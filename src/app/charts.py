# src/app/charts.py
from __future__ import annotations

import plotly.graph_objects as go


def apply_premium_layout(fig: go.Figure, *, height: int = 420, title: str | None = None) -> go.Figure:
    """Apply a consistent premium layout across the app."""

    if title is not None:
        fig.update_layout(title=title)

    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        font=dict(size=12),
        title=dict(font=dict(size=16), x=0),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        ticks="outside",
    )

    return fig


def add_value_labels_bar(fig: go.Figure, *, fmt: str = ",.0f") -> go.Figure:
    """Add simple value labels to bar charts (when bars are few)."""
    for trace in fig.data:
        if getattr(trace, "type", None) == "bar":
            trace.update(texttemplate=f"%{{y:{fmt}}}", textposition="outside", cliponaxis=False)
    fig.update_layout(uniformtext_minsize=10, uniformtext_mode="hide")
    return fig
