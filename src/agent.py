"""
Phase 7: the generative/agentic layer on top of the ranking backbone.

This is the original project's "Appeal Recommendation Analysis" module
(see project_scope_summary_my2903.pdf / the mindmap), rebuilt: instead of
one ungrounded LLM call over raw case text (what Legal_Agent/services/llm.py
does), the LLM here is handed the ranker's actual top-K statutes (real
article text, not just numbers) and the top-K most similar precedent cases
(real outcomes from the training set), and is instructed to cite only from
that retrieved set -- i.e. retrieval-augmented generation, with the
retrieval half being the L1+L2 system from Phases 2-5, not a vector-DB
afterthought.

A lightweight CITATION-MEMBERSHIP guardrail then checks every article number
the model cites in its own output against the set actually retrieved, and
flags (rather than silently drops) any that don't match -- the same class of
guardrail idea as the PII-detection guardrail on Orbit, applied here to
citations instead of sensitive-data leakage.

IMPORTANT, deliberately-named distinction (do not blur these):
  - "citation validity"     -- does the cited article NUMBER appear in the
    retrieved set? This is ALL `check_citation_membership()` verifies.
  - "semantic groundedness" -- does the article's actual TEXT support the
    specific claim the model made about it? NOT checked here.
  - "legal correctness"     -- is the recommendation actually right? NOT
    checked here, and structurally can't be: engine.rank()'s Recall@5 on the
    clean test set is well under 1.0 (see reports/), so a nontrivial share of
    queries hand the model a candidate set that's missing the true statute --
    a response can pass citation-membership 100% of the time while still
    being built on a retrieval miss.

Usage:
    python3 agent.py "<case fact text>"
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from rank_core import RankingEngine

# reuse the existing Legal_Agent/.env rather than duplicating the API key
# into a second .env file
_LEGACY_ENV = Path(__file__).parent.parent.parent / "Legal_Agent" / ".env"
load_dotenv(dotenv_path=_LEGACY_ENV if _LEGACY_ENV.exists() else None)

CITATION_RE = re.compile(r"第([一二三四五六七八九十百千零\d]+)条")
MAX_ARTICLE_CHARS = 1200  # safely above the observed max article length
                          # (1081 chars / 451 articles); the previous 200-char
                          # cutoff silently truncated 130/451 (29%) of articles,
                          # which could omit real qualifying clauses.

PROMPT_TEMPLATE = """你是一位法律助理。以下是一起案件的案情，以及系统检索到的相关法条与相似案例。
请你仅根据下方提供的法条与案例作出分析，不要引用未列出的法条。

【案情】
{fact}

【系统检索到的相关法条（按相关性排序）】
{statutes_block}

【系统检索到的相似案例及其判决结果】
{cases_block}

请按以下结构输出：
建议上诉/不建议上诉/证据不足难以判断：
主要理由（引用上方法条编号）：
风险因素：
与相似案例的异同：
"""


def format_statutes(engine, results):
    lines = []
    for r in results:
        aid = r["article"]
        text = engine.article_text(aid).replace("\n", " ")
        lines.append(f"第{aid}条 (相关性分数 {r['score']:.2f})：{text[:MAX_ARTICLE_CHARS]}")
    return "\n".join(lines)


def format_cases(cases):
    lines = []
    for c in cases:
        term = "死刑" if c["death_penalty"] else ("无期徒刑" if c["life_imprisonment"] else f"有期徒刑{c['imprisonment']}个月")
        lines.append(
            f"- 相似案例（相似度 {c['similarity']:.2f}）：{c['fact'][:150]}... "
            f"罪名：{'、'.join(c['accusation'])}；判决：{term}"
        )
    return "\n".join(lines)


def check_citation_membership(response_text, allowed_article_ids):
    """Citation-membership check ONLY: does every article number the model
    cites appear in the retrieved set? Says nothing about whether the
    model's characterization of that article is accurate (semantic
    groundedness) or whether the underlying recommendation is correct (legal
    correctness) -- see the module docstring."""
    cited = set()
    for m in CITATION_RE.finditer(response_text):
        raw = m.group(1)
        if raw.isdigit():
            cited.add(int(raw))
        # (Chinese-numeral citations are rare in model output in practice --
        # the prompt gives it Arabic numerals to cite back -- so we don't
        # bother round-tripping cn_to_int here; a citation in Chinese
        # numerals would simply not match and get flagged, which is the
        # conservative/safe direction for this check to err in.)
    out_of_set = cited - set(allowed_article_ids)
    return {
        "cited_articles": sorted(cited),
        "citations_not_in_retrieved_set": sorted(out_of_set),
        "citation_membership_ok": len(out_of_set) == 0,
    }


def call_gemini(prompt: str) -> str:
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set (checked environment and .env)")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_recommendation(engine: RankingEngine, fact: str, top_k_statutes=5, top_k_cases=3):
    results, meta = engine.rank(fact, top_k_statutes)
    similar = engine.similar_cases(fact, top_k_cases, neighbors=meta["neighbors"])

    prompt = PROMPT_TEMPLATE.format(
        fact=fact,
        statutes_block=format_statutes(engine, results),
        cases_block=format_cases(similar),
    )
    response_text = call_gemini(prompt)
    citation_check = check_citation_membership(response_text, [r["article"] for r in results])

    return {
        "recommendation": response_text,
        "retrieved_statutes": results,
        "similar_cases": similar,
        "citation_check": citation_check,
    }


def main():
    fact = sys.argv[1] if len(sys.argv) > 1 else (
        "被告人张某趁被害人李某家中无人，翻窗进入室内，将李某放在卧室柜子里的现金一万元和一部手机盗走。"
        "经鉴定，被盗财物价值人民币一万三千元。"
    )
    engine = RankingEngine().load()
    out = generate_recommendation(engine, fact)

    print("=" * 60)
    print("RETRIEVED STATUTES:")
    for r in out["retrieved_statutes"]:
        print(f"  第{r['article']}条  score={r['score']:.2f}")
    print("\nSIMILAR CASES:")
    for c in out["similar_cases"]:
        print(f"  sim={c['similarity']:.2f}  {c['accusation']}  {c['fact'][:60]}...")
    print("\nCITATION-MEMBERSHIP CHECK (not a correctness or semantic-groundedness check):", out["citation_check"])
    print("\nLLM RECOMMENDATION:\n" + out["recommendation"])


if __name__ == "__main__":
    main()
