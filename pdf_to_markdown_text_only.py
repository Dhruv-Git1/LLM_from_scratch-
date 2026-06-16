from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "build-a-large-language-model-from-scratch.raw.txt"
OUT = ROOT / "build-a-large-language-model-from-scratch.md"


def decode_shifted_pdf_text(text: str) -> str:
    keep = {"\n", "\r", "\t", "\f", " "}
    decoded = []
    for ch in text:
        code = ord(ch)
        if ch in keep:
            decoded.append(ch)
        elif code == 3:
            decoded.append(" ")
        elif 1 <= code <= 126:
            decoded.append(chr(code + 29))
        else:
            decoded.append(ch)
    return "".join(decoded)


def repair_text(text: str) -> str:
    replacements = {
        "\x8b": "(c) ",
        "\x93": "+/-",
        "\x97": "micro",
        "Â‹": "(c) ",
        "Â¶": "'",
        "¶": "'",
        "ś": "'",
        "³": '"',
        "´": '"',
        "²": "-",
        "±": "-",
        "î": "x",
        "Ã¼": "u",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[\x7f-\x9f]+", " ", text)
    return text


def normalize_line(line: str) -> str:
    line = line.replace("\t", "    ")
    line = re.sub(r"[ \u00a0]+", " ", line.strip())
    return line


def title_case_front_matter(line: str) -> str:
    small = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "the", "to", "with"}
    words = line.split()
    titled = []
    for i, word in enumerate(words):
        titled.append(word if i and word in small else word.capitalize())
    return " ".join(titled)


def heading_for(line: str) -> str | None:
    lower = line.lower()
    front_matter = {
        "contents",
        "preface",
        "acknowledgments",
        "about this book",
        "about the author",
        "about the cover illustration",
        "about the cover",
        "copyright",
    }
    if lower in front_matter:
        return f"## {title_case_front_matter(lower)}"

    chapter = re.match(r"^(\d+) [A-Z].+", line)
    if chapter and 1 <= int(chapter.group(1)) <= 9:
        return f"## {line}"

    section = re.match(r"^(\d+(?:\.\d+)+) (.+)", line)
    if section:
        depth = min(6, 2 + section.group(1).count("."))
        return f"{'#' * depth} {line}"

    return None


def is_listing(line: str) -> bool:
    return bool(re.match(r"^(Listing|Figure|Table|NOTE|TIP|WARNING)\b", line))


def is_code_like(line: str) -> bool:
    if not line:
        return False
    if len(line) < 90 and re.search(r"\s{2,}", line) and re.search(
        r"(\(|\)|\[|\]|=|:|,|\.|_|\"|'|#|\\|/)", line
    ):
        return True
    if re.match(r"^(import|from|class|def|return|print|try|except)\b", line):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s{2,}\S", line):
        return True
    if re.match(r"^(\[|\{|\}|\)|\])", line):
        return True
    if line.startswith("#"):
        return True
    return False


def append_paragraph(out: list[str], paragraph: list[str]) -> None:
    if not paragraph:
        return
    merged = paragraph[0]
    for nxt in paragraph[1:]:
        if merged.endswith("-") and nxt[:1].islower():
            merged = merged[:-1] + nxt
        else:
            merged += " " + nxt
    out.append(merged)
    out.append("")
    paragraph.clear()


def convert_to_markdown(text: str) -> str:
    lines = text.splitlines()
    out = [
        "# Build a Large Language Model From Scratch",
        "",
        "Sebastian Raschka",
        "",
        "> Text-only Markdown conversion from the local PDF. Images were omitted. Some code punctuation may need verification against the original source because the PDF text map collapses a few symbols during extraction.",
        "",
    ]
    paragraph: list[str] = []
    in_code = False
    skipped_title = False
    code_after_listing = False

    for raw in lines:
        code_line = raw.rstrip().replace("\t", "    ")
        line = normalize_line(raw)

        if not line or line.isdigit():
            if code_after_listing and in_code:
                out.append("")
                continue
            if not code_after_listing:
                append_paragraph(out, paragraph)
                if in_code:
                    out.append("```")
                    out.append("")
                    in_code = False
            continue

        if not skipped_title and line in {
            "Build a Large Language Model",
            "From Scratch",
            "Sebastian Raschka",
        }:
            continue
        skipped_title = True

        if code_after_listing and in_code:
            is_annotation = bool(re.match(r"^#\d+\s+", line))
            still_code = is_code_like(line) or raw.startswith((" ", "\t"))
            if is_annotation or not still_code:
                out.append("```")
                out.append("")
                in_code = False
                code_after_listing = False

        heading = heading_for(line)
        if heading:
            append_paragraph(out, paragraph)
            if in_code:
                out.append("```")
                out.append("")
                in_code = False
            out.append(heading)
            out.append("")
            continue

        if is_listing(line):
            append_paragraph(out, paragraph)
            if in_code:
                out.append("```")
                out.append("")
                in_code = False
            out.append(f"**{line}**")
            out.append("")
            if line.startswith("Listing "):
                code_after_listing = True
            continue

        annotation = re.match(r"^#(\d+)\s+(.+)", line)
        if annotation:
            append_paragraph(out, paragraph)
            if in_code:
                out.append("```")
                out.append("")
                in_code = False
            out.append(f"- `{annotation.group(1)}` {annotation.group(2)}")
            continue

        if code_after_listing or is_code_like(line):
            append_paragraph(out, paragraph)
            if not in_code:
                out.append("```python")
                in_code = True
            out.append(code_line if code_line.strip() else line)
            continue

        if in_code:
            out.append("```")
            out.append("")
            in_code = False

        paragraph.append(line)

    append_paragraph(out, paragraph)
    if in_code:
        out.append("```")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    raw_text = RAW.read_text(encoding="utf-8", errors="replace")
    decoded = repair_text(decode_shifted_pdf_text(raw_text))
    OUT.write_text(convert_to_markdown(decoded), encoding="utf-8")


if __name__ == "__main__":
    main()
