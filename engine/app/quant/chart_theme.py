"""Shared Plotly styling so every generated chart looks like one system."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

PRESETS: dict[str, dict[str, Any]] = {
    "professional": {
        "bg": "#ffffff", "grid": "#e6e9ef", "font": "#1a1f2b", "accent": "#2563eb",
        "palette": ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#65a30d"],
    },
    "dark_quant": {
        "bg": "#0b0e14", "grid": "#232838", "font": "#e6e9ef", "accent": "#5b9dff",
        "palette": ["#5b9dff", "#ff6b6b", "#2dd4bf", "#fbbf24", "#a78bfa", "#22d3ee", "#f472b6", "#a3e635"],
    },
    "minimal": {
        "bg": "#ffffff", "grid": "#f0f0f0", "font": "#222222", "accent": "#111111",
        "palette": ["#111111", "#888888", "#c0392b", "#2c7fb8", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    },
    "bloomberg": {
        "bg": "#000000", "grid": "#1a1a00", "font": "#ff8c00", "accent": "#ff8c00",
        "palette": ["#ff8c00", "#ffffff", "#00d5ff", "#ffee00", "#ff4d4d", "#7cfc00", "#c299ff", "#40e0d0"],
    },
    "presentation": {
        "bg": "#fbfbfd", "grid": "#e3e7ee", "font": "#111827", "accent": "#4f46e5",
        "palette": ["#4f46e5", "#ea580c", "#0d9488", "#dc2626", "#0284c7", "#65a30d", "#c026d3", "#78716c"],
    },
}

DEFAULT_PRESET = "professional"


def apply_theme(
    fig: go.Figure,
    *,
    preset: str = DEFAULT_PRESET,
    title: str | None = None,
    subtitle: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    font_size: int = 13,
    height: int = 480,
) -> go.Figure:
    theme = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    full_title = title or ""
    if subtitle:
        full_title = f"{title}<br><span style='font-size:0.75em;color:{theme['font']}99'>{subtitle}</span>"

    # A subtitle makes the title block two lines instead of one — the fixed
    # single-line top margin (70px) isn't tall enough for that second line,
    # so the plot area's top edge/gridlines crept up into the subtitle text.
    # Scale the margin with subtitle length as a rough proxy for how many
    # lines it'll wrap to at this font size (long subtitles, e.g. ones
    # listing several security names, wrap to 2-3 lines on a typical chart
    # width). Callers with unusual layouts can still override margin after
    # calling this.
    if subtitle:
        wrapped_lines = 1 + len(subtitle) // 70
        top_margin = 70 + 22 * wrapped_lines
    elif title:
        top_margin = 70
    else:
        top_margin = 30

    fig.update_layout(
        template=None,
        paper_bgcolor=theme["bg"],
        plot_bgcolor=theme["bg"],
        font=dict(color=theme["font"], size=font_size, family="Inter, -apple-system, Helvetica, Arial, sans-serif"),
        title=dict(text=full_title, x=0.02, xanchor="left", font=dict(size=font_size + 5)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=30, t=top_margin, b=50),
        height=height,
        hovermode="x unified",
        colorway=theme["palette"],
    )
    fig.update_xaxes(title=x_title, gridcolor=theme["grid"], zerolinecolor=theme["grid"], showline=True,
                      linecolor=theme["grid"])
    fig.update_yaxes(title=y_title, gridcolor=theme["grid"], zerolinecolor=theme["grid"], showline=True,
                      linecolor=theme["grid"])
    return fig
