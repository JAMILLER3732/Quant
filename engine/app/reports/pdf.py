"""
Renders a ReportData (+ generated narrative) into a professional,
investment-bank-style HTML document, then to PDF via WeasyPrint.

Design references: institutional equity research notes (e.g. sell-side
single-stock notes) and quant/portfolio performance reports — serif
headline type, a thin rule under the masthead, a compact stats grid,
one chart per section rendered at print resolution, and a footer
disclaimer on every page. Charts are rendered to static PNG via Kaleido
so they embed cleanly in the PDF (WeasyPrint doesn't run Plotly's JS).
"""
from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from app.reports.builder import ReportData

DISCLAIMER = (
    "This report was generated automatically from the uploaded dataset using documented, "
    "testable quantitative methods (see Methodology under each section). It describes historical "
    "and statistical characteristics of the data only. It is not investment advice, is not a "
    "recommendation to buy or sell any security, and no statement in this report should be read as "
    "a guarantee or prediction of future performance."
)

CSS = """
@page { size: Letter; margin: 2.2cm 1.8cm; @bottom-center { content: element(footer); } }
* { box-sizing: border-box; }
body {
  font-family: Georgia, 'Times New Roman', serif;
  color: #1a1f2b;
  font-size: 10.5pt;
  line-height: 1.5;
}
.masthead { border-bottom: 3px solid #111827; padding-bottom: 10px; margin-bottom: 4px; }
.masthead .kicker { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; letter-spacing: 0.12em;
  text-transform: uppercase; color: #6b7280; margin: 0 0 6px 0; }
.masthead h1 { font-size: 22pt; margin: 0 0 4px 0; font-weight: 700; }
.masthead .subtitle { font-size: 11pt; color: #4b5563; margin: 0; font-style: italic; }
.meta-row { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: #6b7280;
  display: flex; justify-content: space-between; margin: 8px 0 20px 0; border-bottom: 1px solid #d1d5db;
  padding-bottom: 8px; }
.disclaimer-box { background: #f9fafb; border-left: 3px solid #9ca3af; padding: 8px 12px;
  font-family: Helvetica, Arial, sans-serif; font-size: 8pt; color: #4b5563; margin-bottom: 22px; }
h2.section-title { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; letter-spacing: 0.08em;
  text-transform: uppercase; color: #111827; border-bottom: 1.5px solid #111827; padding-bottom: 4px;
  margin: 26px 0 10px 0; }
h3.method-title { font-size: 13pt; margin: 18px 0 4px 0; }
p { margin: 6px 0; text-align: justify; }
.stats-grid { display: flex; flex-wrap: wrap; gap: 0; margin: 10px 0 14px 0; border-top: 1px solid #e5e7eb;
  border-left: 1px solid #e5e7eb; }
.stat-tile { width: 25%; border-right: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb;
  padding: 8px 10px; font-family: Helvetica, Arial, sans-serif; }
.stat-tile .label { font-size: 7.5pt; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }
.stat-tile .value { font-size: 12pt; font-weight: 700; color: #111827; margin-top: 2px; }
.chart-img { width: 100%; margin: 10px 0 4px 0; border: 1px solid #e5e7eb; }
.warning-box { font-family: Helvetica, Arial, sans-serif; font-size: 8pt; color: #92400e; background: #fffbeb;
  border-left: 3px solid #f59e0b; padding: 6px 10px; margin: 8px 0; }
.methodology { font-family: Helvetica, Arial, sans-serif; font-size: 8.5pt; color: #374151; background: #f9fafb;
  padding: 8px 10px; margin: 8px 0; }
.methodology .label { font-weight: 700; text-transform: uppercase; font-size: 7.5pt; letter-spacing: 0.05em;
  color: #6b7280; display: block; margin-bottom: 3px; }
table.ranking { width: 100%; border-collapse: collapse; font-family: Helvetica, Arial, sans-serif; font-size: 8pt;
  margin: 8px 0; }
table.ranking th { text-align: left; border-bottom: 1.5px solid #111827; padding: 4px 6px; }
table.ranking td { border-bottom: 1px solid #e5e7eb; padding: 4px 6px; }
.footer { font-family: Helvetica, Arial, sans-serif; font-size: 7pt; color: #9ca3af; text-align: center; }
.ai-badge { font-family: Helvetica, Arial, sans-serif; font-size: 7pt; color: #6b7280; font-style: italic; }
"""


def _fig_to_png_base64(fig_dict: dict[str, Any], width: int = 1000, height: int = 460) -> str:
    fig = go.Figure(fig_dict)
    png_bytes = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return base64.b64encode(png_bytes).decode("ascii")


def _stats_grid_html(stats: dict[str, Any]) -> str:
    tiles = []
    for label, value in stats.items():
        tiles.append(
            f'<div class="stat-tile"><div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{html.escape(str(value))}</div></div>'
        )
    return f'<div class="stats-grid">{"".join(tiles)}</div>'


def _ranking_table_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<table class="ranking"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def render_report_html(data: ReportData, narrative: dict[str, str]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
    date_range_str = f"{data.date_range[0]} – {data.date_range[1]}" if data.date_range else "N/A"

    sections_html = []
    for section in data.sections:
        method = None
        try:
            from app.quant.registry import get_method
            method = get_method(section.method_id)
        except Exception:  # noqa: BLE001
            pass

        chart_html = ""
        if section.figure:
            png_b64 = _fig_to_png_base64(section.figure)
            chart_html = f'<img class="chart-img" src="data:image/png;base64,{png_b64}" />'

        warnings_html = "".join(
            f'<div class="warning-box">⚠ {html.escape(w)}</div>' for w in section.warnings
        )

        methodology_html = ""
        if method:
            methodology_html = (
                f'<div class="methodology"><span class="label">Methodology</span>{html.escape(method.methodology)}</div>'
            )

        extra_tables = ""
        for name, table in section.tables.items():
            if isinstance(table, list) and table and isinstance(table[0], dict):
                extra_tables += f'<h3 class="method-title" style="font-size:10.5pt">{html.escape(name.replace("_", " ").title())}</h3>'
                extra_tables += _ranking_table_html(table)

        sections_html.append(f"""
        <h3 class="method-title">{html.escape(section.method_name)}</h3>
        {warnings_html}
        {_stats_grid_html(section.stats)}
        {chart_html}
        {extra_tables}
        {methodology_html}
        """)

    ai_note = (
        f'<span class="ai-badge">Narrative sections were AI-generated (grounded in the computed figures above) '
        f'via {narrative.get("model", "an LLM")}.</span>'
        if narrative.get("ai_generated") == "true"
        else '<span class="ai-badge">Narrative sections are templated from the computed figures (no AI narrative configured).</span>'
    )

    return f"""
    <style>{CSS}</style>
    <div class="footer" style="position: running(footer);">
      {html.escape(DISCLAIMER)} &nbsp;•&nbsp; Generated {generated_at}
    </div>
    <div class="masthead">
      <p class="kicker">Quant Analytics Platform — Research Note</p>
      <h1>{html.escape(data.title)}</h1>
      <p class="subtitle">{html.escape(data.subtitle)}</p>
    </div>
    <div class="meta-row">
      <span>Coverage: {html.escape(data.target)}</span>
      <span>Period: {html.escape(date_range_str)}</span>
      <span>Observations: {data.n_observations}</span>
      <span>Generated: {generated_at}</span>
    </div>
    <div class="disclaimer-box">{html.escape(DISCLAIMER)}</div>

    <h2 class="section-title">Executive Summary</h2>
    <p>{html.escape(narrative.get("executive_summary", ""))}</p>

    <h2 class="section-title">Quantitative Analysis</h2>
    <p>{html.escape(narrative.get("analysis", ""))}</p>
    {"".join(sections_html)}

    <h2 class="section-title">Assumptions &amp; Limitations</h2>
    <p>{html.escape(narrative.get("limitations", ""))}</p>
    {"".join(f'<div class="warning-box">⚠ {html.escape(w)}</div>' for w in data.warnings)}

    <p style="margin-top: 20px;">{ai_note}</p>
    """


def render_report_pdf(data: ReportData, narrative: dict[str, str]) -> bytes:
    from weasyprint import HTML

    html_content = render_report_html(data, narrative)
    return HTML(string=f"<html><body>{html_content}</body></html>").write_pdf()
