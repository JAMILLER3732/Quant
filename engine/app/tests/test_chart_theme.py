"""
Regression test for a chart-header overlap bug: apply_theme() used a fixed
70px top margin regardless of whether a subtitle was present. A subtitle
turns the title block into two (or more) lines, and 70px wasn't tall enough
for that — the plot area's top edge/gridlines crept up into the subtitle
text on every chart that sets one (correlation heatmap, stress test,
performance dashboard's "many securities" note, ...). The margin must now
scale with subtitle length instead of being a fixed constant.
"""
from __future__ import annotations

import plotly.graph_objects as go

from app.quant.chart_theme import apply_theme


def test_no_title_gets_small_margin():
    fig = apply_theme(go.Figure())
    assert fig.layout.margin.t == 30


def test_title_only_gets_standard_margin():
    fig = apply_theme(go.Figure(), title="Some Chart")
    assert fig.layout.margin.t == 70


def test_short_subtitle_gets_more_margin_than_title_alone():
    fig = apply_theme(go.Figure(), title="Some Chart", subtitle="A short note")
    assert fig.layout.margin.t > 70


def test_long_subtitle_gets_even_more_margin_than_short_subtitle():
    short = apply_theme(go.Figure(), title="Some Chart", subtitle="short")
    long = apply_theme(
        go.Figure(), title="Some Chart",
        subtitle="AVGO, GEV, NCLH, NVDA, PTON, RDDT, RUM, SEDG, SHOP, TSLA had an implied shock beyond -100%",
    )
    assert long.layout.margin.t > short.layout.margin.t
