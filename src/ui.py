"""Shared colours, page styling, and chart helpers for the Streamlit interface.

Brand colours mirror .streamlit/config.toml. Data colours follow a validated palette:
sentiment uses a blue/gray/red diverging set and magnitudes use a single blue ramp.
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

# Brand chrome (keep in sync with .streamlit/config.toml).
INK = "#1C2B33"
MUTED = "#5B6870"
PAPER = "#F6F5F1"
SURFACE = "#FFFFFF"
BORDER = "#DDD9CE"
GRID = "#E9E6DE"
TEAL = "#0E7C7B"
TEAL_DARK = "#12262E"
AMBER = "#F2A541"

# Data colours. Sentiment is polarity, so it uses the diverging poles with a gray midpoint.
SENTIMENT_COLORS = {"positive": "#2a78d6", "neutral": "#8a8880", "negative": "#e34948"}
SENTIMENT_ICONS = {"positive": "▲", "neutral": "●", "negative": "▼"}
SERIES_BLUE = "#2a78d6"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_CSS = f"""
<style>
/* Leave room for Streamlit's fixed top toolbar (Share, GitHub, etc.) so the brand header isn't hidden behind it. */
.block-container {{ padding-top: 4.25rem; padding-bottom: 3rem; max-width: 1240px; }}

/* Example chips on the single-comment page */
.st-key-examples button {{ border-radius: 999px; padding: 0.25rem 0.9rem; min-height: 0; background: {SURFACE}; }}
.st-key-examples button:hover {{ border-color: {TEAL}; color: {TEAL}; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* Brand bar */
.brand {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }}
.brand-left {{ display: flex; align-items: center; gap: 0.8rem; }}
.brand-mark {{ width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, {TEAL} 0%, {TEAL_DARK} 100%); box-shadow: 0 6px 16px rgba(14,124,123,0.25); }}
.brand-name {{ font-family: "Source Serif", "Source Serif Pro", Georgia, serif; font-size: 1.35rem; font-weight: 700; color: {INK}; line-height: 1.1; }}
.brand-tag {{ color: {MUTED}; font-size: 0.86rem; }}
.status-chip {{ display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.8rem; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; border: 1px solid {BORDER}; background: {SURFACE}; color: {INK}; }}
.status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: #B8B4A8; }}
.status-chip.ready .status-dot {{ background: {TEAL}; box-shadow: 0 0 0 3px rgba(14,124,123,0.18); }}

/* Navigation: the radio group rendered as underlined tabs */
.st-key-nav div[role="radiogroup"] {{ display: flex; flex-wrap: wrap; gap: 0.25rem; border-bottom: 1px solid {BORDER};
  margin: 0.6rem 0 0.4rem; }}
.st-key-nav div[role="radiogroup"] label {{ padding: 0.55rem 0.85rem; margin: 0 0 -1px 0; border-bottom: 3px solid transparent;
  border-radius: 8px 8px 0 0; transition: background 120ms ease, border-color 120ms ease; }}
.st-key-nav div[role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display: none; }}
.st-key-nav div[role="radiogroup"] label p {{ font-weight: 600; color: {MUTED}; font-size: 0.95rem; }}
.st-key-nav div[role="radiogroup"] label:hover {{ background: rgba(14,124,123,0.06); }}
.st-key-nav div[role="radiogroup"] label[data-selected="true"] {{ border-bottom-color: {TEAL}; }}
.st-key-nav div[role="radiogroup"] label[data-selected="true"] p {{ color: {INK}; }}
.st-key-nav div[role="radiogroup"] label:has(input:focus-visible) {{ outline: 2px solid {TEAL}; outline-offset: 2px; }}

/* Workflow stepper */
.stepper {{ display: flex; gap: 0.5rem; margin: 0.4rem 0 1.4rem; flex-wrap: wrap; }}
.step {{ flex: 1 1 180px; display: flex; align-items: center; gap: 0.6rem; padding: 0.6rem 0.8rem; border-radius: 12px;
  border: 1px solid {BORDER}; background: {SURFACE}; }}
.step-num {{ width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 0.85rem;
  background: {GRID}; color: {MUTED}; flex-shrink: 0; }}
.step-title {{ font-weight: 650; font-size: 0.92rem; color: {INK}; line-height: 1.15; }}
.step-hint {{ font-size: 0.78rem; color: {MUTED}; }}
.step.done .step-num {{ background: {TEAL}; color: #fff; }}
.step.current {{ border-color: {TEAL}; box-shadow: 0 0 0 3px rgba(14,124,123,0.12); }}
.step.current .step-num {{ background: #fff; color: {TEAL}; box-shadow: inset 0 0 0 2px {TEAL}; }}

/* Section headers */
.eyebrow {{ text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem; font-weight: 700; color: {TEAL}; margin-bottom: 0.15rem; }}
.section-lead {{ color: {MUTED}; margin: -0.4rem 0 1rem; font-size: 0.98rem; }}

/* Hero */
.st-key-hero {{ background: radial-gradient(circle at 85% 15%, rgba(242,165,65,0.35) 0%, rgba(242,165,65,0) 40%),
  linear-gradient(135deg, {TEAL_DARK} 0%, #0E4D52 55%, {TEAL} 100%); border-radius: 20px; padding: clamp(1.4rem, 2.6vw, 2.4rem); }}
.st-key-hero h1, .st-key-hero p, .st-key-hero span, .st-key-hero li {{ color: #F4F8F8; }}
.hero-title {{ font-family: "Source Serif", "Source Serif Pro", Georgia, serif; font-size: clamp(1.6rem, 2.6vw, 2.3rem); font-weight: 700; line-height: 1.15;
  color: #fff; margin: 0 0 0.6rem; max-width: 760px; }}
.hero-title em {{ font-style: normal; color: {AMBER}; }}
.hero-sub {{ font-size: 1.05rem; color: #D5E4E4; max-width: 680px; margin-bottom: 1.2rem; }}
.st-key-hero button[kind="primary"] {{ background: {AMBER}; border-color: {AMBER}; color: {INK}; font-weight: 700; }}
.st-key-hero button[kind="primary"]:hover {{ background: #F5B661; border-color: #F5B661; color: {INK}; }}
.st-key-hero button[kind="secondary"] {{ background: transparent; border-color: rgba(255,255,255,0.55); }}
.st-key-hero button[kind="secondary"] p {{ color: #fff; }}
.st-key-hero button[kind="secondary"]:hover {{ border-color: #fff; background: rgba(255,255,255,0.08); }}

/* Cards */
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%; }}
.card h4 {{ margin: 0.5rem 0 0.25rem; font-size: 1.02rem; }}
.card p {{ color: {MUTED}; font-size: 0.92rem; margin: 0; }}
.card-num {{ width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-weight: 700;
  background: rgba(14,124,123,0.1); color: {TEAL}; }}

/* Single-comment result */
.result {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 6px solid var(--accent); border-radius: 14px; padding: 1rem 1.2rem; height: 100%; }}
.result-label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; font-weight: 700; }}
.result-value {{ font-size: 1.6rem; font-weight: 700; color: {INK}; margin: 0.2rem 0 0.5rem; }}
.result-icon {{ color: var(--accent); margin-right: 0.35rem; }}
.meter {{ height: 8px; border-radius: 999px; background: {GRID}; overflow: hidden; }}
.meter > span {{ display: block; height: 100%; border-radius: 999px; background: var(--accent); }}
.result-note {{ font-size: 0.85rem; color: {MUTED}; margin-top: 0.4rem; }}

/* Primary buttons get a little more presence */
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{ font-weight: 700; box-shadow: 0 6px 14px rgba(14,124,123,0.22); }}
.stMain [data-testid="stMetric"] {{ background: {SURFACE}; }}
[data-testid="stSidebar"] .sidebar-brand {{ font-family: "Source Serif", "Source Serif Pro", Georgia, serif; font-size: 1.15rem; font-weight: 700; color: #fff; }}
[data-testid="stSidebar"] .sidebar-note {{ font-size: 0.82rem; color: #A9C0C5; }}
</style>
"""

_BRAND_ICON = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z"/><path d="M8.8 12.2l2.2 2.2 4.2-4.4"/></svg>'
)


def inject_css() -> None:
    """Apply the shared page styles."""
    st.markdown(_CSS, unsafe_allow_html=True)


def brand_bar(status_text: str, ready: bool) -> None:
    """Render the product name with a data-status chip."""
    chip_class = "status-chip ready" if ready else "status-chip"
    st.markdown(
        f"""
        <div class="brand">
          <div class="brand-left">
            <div class="brand-mark">{_BRAND_ICON}</div>
            <div><div class="brand-name">Quality &amp; Returns Intelligence</div>
            <div class="brand-tag">Customer feedback, defect signals, and return risk in one place</div></div>
          </div>
          <div class="{chip_class}"><span class="status-dot"></span>{escape(status_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stepper(steps: list[tuple[str, str, str]]) -> None:
    """Render workflow steps as (title, hint, state) where state is "done", "current", or "todo"."""
    items = []
    for number, (title, hint, state) in enumerate(steps, start=1):
        marker = "✓" if state == "done" else str(number)
        items.append(
            f'<div class="step {state}"><div class="step-num">{marker}</div>'
            f'<div><div class="step-title">{escape(title)}</div><div class="step-hint">{escape(hint)}</div></div></div>'
        )
    st.markdown(f'<div class="stepper">{"".join(items)}</div>', unsafe_allow_html=True)


def section_header(eyebrow: str, title: str, lead: str | None = None) -> None:
    """Render a small eyebrow label, a heading, and an optional lead sentence."""
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if lead:
        st.markdown(f'<div class="section-lead">{escape(lead)}</div>', unsafe_allow_html=True)


def card(number: str, title: str, body: str) -> None:
    """Render a numbered explainer card."""
    st.markdown(
        f'<div class="card"><div class="card-num">{escape(number)}</div><h4>{escape(title)}</h4><p>{escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def result_card(label: str, value: str, accent: str, icon: str = "", meter: float | None = None, note: str = "") -> None:
    """Render a single result with an accent colour, an optional icon, and an optional 0-1 meter."""
    icon_html = f'<span class="result-icon" aria-hidden="true">{escape(icon)}</span>' if icon else ""
    meter_html = f'<div class="meter"><span style="width:{max(0.0, min(meter, 1.0)) * 100:.0f}%"></span></div>' if meter is not None else ""
    note_html = f'<div class="result-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="result" style="--accent:{accent}"><div class="result-label">{escape(label)}</div>'
        f'<div class="result-value">{icon_html}{escape(value)}</div>{meter_html}{note_html}</div>',
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = False) -> go.Figure:
    """Apply the shared chart look: quiet grid, thin rounded bars, readable hover labels."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "family": "Source Serif, Source Serif Pro, Georgia, serif", "color": INK}}
        if title
        else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Source Sans, Source Sans Pro, sans-serif", "size": 13, "color": INK},
        margin={"l": 8, "r": 16, "t": 52 if title else 12, "b": 8},
        hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"color": INK, "size": 13}},
        showlegend=legend,
        barcornerradius=4,
        bargap=0.5,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", title_font={"color": MUTED, "size": 12})
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="", title_font={"color": MUTED, "size": 12})
    return fig


def heatmap_scale() -> list[list]:
    """Blue sequential scale whose zero end fades into the page, so empty cells recede."""
    steps = ["#F1F5FB"] + SEQUENTIAL_BLUE[1:]
    return [[index / (len(steps) - 1), color] for index, color in enumerate(steps)]


def show_chart(fig: go.Figure) -> None:
    """Render a Plotly figure with the shared styling kept intact."""
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
