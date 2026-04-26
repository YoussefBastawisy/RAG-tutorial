"""Render docs/workshop_guide.md to docs/workshop_guide.pdf.

Run from the project root:
    python docs/_build_pdf.py

Requires: pip install markdown-pdf
"""
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

DOCS_DIR = Path(__file__).parent
MD_PATH = DOCS_DIR / "workshop_guide.md"
PDF_PATH = DOCS_DIR / "workshop_guide.pdf"

CSS = """
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; line-height: 1.55; color: #222; }
h1 { font-size: 26pt; color: #1a1a1a; border-bottom: 2px solid #444; padding-bottom: 6pt; margin-top: 18pt; }
h2 { font-size: 18pt; color: #2a2a2a; border-bottom: 1px solid #999; padding-bottom: 4pt; margin-top: 16pt; }
h3 { font-size: 14pt; color: #333; margin-top: 14pt; }
h4 { font-size: 12pt; color: #444; margin-top: 12pt; }
p, li { font-size: 10.5pt; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-family: 'Consolas', 'Courier New', monospace; font-size: 9.5pt; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: 8pt; border-radius: 4pt; font-size: 9pt; overflow-x: auto; }
pre code { background: transparent; padding: 0; }
blockquote { border-left: 3px solid #888; padding: 2pt 10pt; color: #444; background: #fafafa; margin: 8pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 8pt 0; }
th, td { border: 1px solid #bbb; padding: 5pt 7pt; text-align: left; vertical-align: top; }
th { background: #eee; }
hr { border: none; border-top: 1px solid #ccc; margin: 14pt 0; }
"""


def main() -> None:
    if not MD_PATH.exists():
        raise SystemExit(f"Markdown source not found: {MD_PATH}")

    md = MD_PATH.read_text(encoding="utf-8")

    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.meta["title"] = "Build a RAG-Powered Customer Support Agent"
    pdf.meta["author"] = "Eng. Youssef Bastawisy"
    pdf.meta["subject"] = "3-hour workshop guide"
    pdf.meta["keywords"] = "RAG, LangGraph, LangChain, Streamlit, Groq, Chroma, workshop"

    pdf.add_section(Section(md, toc=True), user_css=CSS)
    pdf.save(str(PDF_PATH))

    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"Wrote {PDF_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
