"""Render a markdown report to PDF.

Pure-Python conversion (markdown -> HTML -> PDF via xhtml2pdf), so no
external binaries (pandoc, wkhtmltopdf, ...) are required. Images referenced
with relative paths in the markdown are resolved relative to the markdown
file's own directory.

Run:  python scripts/report_to_pdf.py [src.md] [dest.pdf]
Defaults: docs/report.md -> docs/report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]

_CSS = """
<style>
  @page { size: A4; margin: 1.6cm; }
  body { font-family: Helvetica, sans-serif; font-size: 9.5pt; line-height: 1.35; }
  h1 { font-size: 17pt; margin-top: 0; }
  h2 { font-size: 12.5pt; margin-top: 12pt; border-bottom: 1px solid #ccc; }
  h3 { font-size: 10.5pt; margin-top: 8pt; }
  code, pre { font-family: Courier, monospace; font-size: 8pt; }
  pre { background: #f4f4f4; padding: 6px; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0; }
  th, td { border: 1px solid #999; padding: 3px 5px; font-size: 8.5pt; text-align: left; }
  th { background: #eee; }
  a { color: #1a5276; }
  img { max-width: 100%; }
</style>
"""


def _link_callback(uri: str, rel: str, base: Path) -> str:
    """Resolve relative image paths (used by xhtml2pdf for <img src=...>)."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    path = (base / uri).resolve()
    return str(path)


def main() -> int:
    argv = sys.argv[1:]
    src = ROOT / (argv[0] if len(argv) > 0 else "docs/report.md")
    dest = ROOT / (argv[1] if len(argv) > 1 else "docs/report.pdf")

    md_text = src.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        md_text, extensions=["extra", "sane_lists", "toc"]
    )
    html = f"<html><head>{_CSS}</head><body>{body_html}</body></html>"

    with open(dest, "wb") as fh:
        result = pisa.CreatePDF(
            html, dest=fh, link_callback=lambda uri, rel: _link_callback(uri, rel, src.parent)
        )
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s).")

    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

