"""Render a markdown report to PDF.

Pure-Python conversion (markdown -> HTML -> PDF via xhtml2pdf), so no
external binaries (pandoc, wkhtmltopdf, ...) are required. Images referenced
with relative paths in the markdown are resolved relative to the markdown
file's own directory.

Run:  python scripts/report_to_pdf.py [src.md] [dest.pdf]
Defaults: docs/report.md -> docs/report.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
_MD_EXTENSIONS = ["extra", "sane_lists", "toc"]

_CSS = """
<style>
  @page { size: A4; margin: 1.6cm; }
  body { font-family: Helvetica, sans-serif; font-size: 9.5pt; line-height: 1.35; }
  h1 { font-size: 17pt; margin-top: 0; }
  h2 { font-size: 12.5pt; margin-top: 12pt; border-bottom: 1px solid #ccc; }
  h3 { font-size: 10.5pt; margin-top: 8pt; }
  code, pre { font-family: Courier, monospace; font-size: 8pt; }
  pre { background: #f4f4f4; padding: 6px; }
  table { border-collapse: collapse; width: 100%; margin: 6px 0; table-layout: fixed; }
  th, td { border: 1px solid #999; padding: 3px 5px; font-size: 8.5pt; text-align: left;
           word-wrap: break-word; overflow-wrap: break-word; }
  th { background: #eee; }
  a { color: #1a5276; }
  img { max-width: 100%; }
</style>
"""


def _link_callback(uri: str, rel: str, base: Path) -> str:
    """Resolve relative image paths (used by xhtml2pdf for <img src=...>)."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    return str((base / uri).resolve())


def render_html(md_text: str) -> str:
    """Convert report markdown into a standalone, styled HTML document."""
    body_html = markdown.markdown(md_text, extensions=_MD_EXTENSIONS)
    return f"<html><head>{_CSS}</head><body>{body_html}</body></html>"


def render_pdf(html: str, dest: Path, image_base_dir: Path) -> None:
    """Render an HTML document to ``dest`` via xhtml2pdf.

    ``image_base_dir`` is where relative <img src=...> paths are resolved
    from -- the source markdown file's own directory, not the CWD.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        result = pisa.CreatePDF(
            html,
            dest=fh,
            link_callback=lambda uri, rel: _link_callback(uri, rel, image_base_dir),
        )
    if result.err:
        raise RuntimeError(f"PDF generation failed with {result.err} error(s).")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "src", nargs="?", default="docs/report.md",
        help="Source markdown file (relative to repo root). Default: docs/report.md",
    )
    p.add_argument(
        "dest", nargs="?", default=None,
        help="Destination PDF path (relative to repo root). Default: <src> with a .pdf suffix.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    src = ROOT / args.src
    dest = ROOT / args.dest if args.dest else src.with_suffix(".pdf")

    html = render_html(src.read_text(encoding="utf-8"))
    render_pdf(html, dest, image_base_dir=src.parent)

    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

