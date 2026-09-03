"""
Phase 1a: Parse the full text of the PRC Criminal Law into a per-article
corpus: {article_number: {"header": crime name, "text": full article text}}.

Source: data/statutes/raw_criminal_law_chnlawyer.html (2024-revised text,
through Amendment XII). This is the "document corpus" our retrieval/ranking
system will search over -- analogous to a product catalog in e-commerce
search.

Caveat (documented, not hidden): CAIL2018 case labels were annotated against
an earlier revision of the Criminal Law (pre-Amendment XI/XII). Amendments in
Chinese criminal law almost always *add* sub-articles (e.g. 133条之一)
rather than renumbering existing ones, so article IDs 1-452 referenced by
CAIL should still resolve correctly against this newer text. We do not
attempt to reconstruct the exact pre-amendment wording.
"""
import json
import re
from pathlib import Path

RAW_HTML = Path(__file__).parent.parent / "data/statutes/raw_criminal_law_chnlawyer.html"
OUT_JSON = Path(__file__).parent.parent / "data/statutes/articles.json"

CN_NUM = "零一二三四五六七八九十百千"
CN_DIGIT = {c: i for i, c in enumerate("零一二三四五六七八九")}


def cn_to_int(s: str) -> int:
    """Convert a Chinese numeral string (e.g. '四百五十二') to an int."""
    if not s:
        return 0
    total, section, unit_val = 0, 0, 1
    units = {"十": 10, "百": 100, "千": 1000}
    i = 0
    num = 0
    for ch in s:
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
        elif ch in units:
            u = units[ch]
            if num == 0:
                num = 1
            section += num * u
            num = 0
        else:
            continue
    section += num
    return section


ARTICLE_START = re.compile(
    r"^第([一二三四五六七八九十百千零]+)条(之[一二三四五六七八九十]+)?[　\s](.*)$"
)
HEADER = re.compile(r"^【([^】]+)】(.*)$")


def html_to_lines(html: str):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = text.replace("&nbsp;", " ")
    return [l.strip() for l in text.split("\n") if l.strip()]


def parse_articles():
    html = RAW_HTML.read_text(encoding="utf-8", errors="ignore")
    lines = html_to_lines(html)

    starts = [i for i, l in enumerate(lines) if ARTICLE_START.match(l)]
    if not starts:
        raise RuntimeError("No article markers found -- source HTML structure changed.")

    # trim to the actual body: first start .. last start's article block
    articles = {}
    for idx, line_no in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else line_no + 1
        block_lines = [lines[line_no]] + lines[line_no + 1 : end]
        # only keep continuation lines that don't look like nav/footer junk
        block_lines = [b for b in block_lines if len(b) > 1][:40]
        full_text = "\n".join(block_lines)

        m = ARTICLE_START.match(lines[line_no])
        cn_num, suffix, rest = m.group(1), m.group(2), m.group(3)
        art_no = cn_to_int(cn_num)
        key = f"{art_no}{suffix.replace('之', '_') if suffix else ''}"

        # the 【crime name】 bracket sometimes wraps onto the next raw line
        # (the site's <p> markup splits mid-bracket); search the joined
        # first two lines instead of just `rest`.
        header_search_text = rest + (block_lines[1] if len(block_lines) > 1 else "")
        hm = HEADER.match(header_search_text) or re.search(r"【([^】]+)】", header_search_text)
        header = hm.group(1) if hm else None

        articles[key] = {
            "article_no": art_no,
            "sub": suffix,
            "header": header,
            "text": full_text,
        }
    return articles


def main():
    articles = parse_articles()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(articles, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    base_nums = {v["article_no"] for v in articles.values()}
    print(f"Parsed {len(articles)} article entries covering {len(base_nums)} base article numbers "
          f"(range {min(base_nums)}-{max(base_nums)})")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
