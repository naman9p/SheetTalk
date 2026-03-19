import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from openai import OpenAI
from datetime import datetime

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="SheetTalk — Chat With Your Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SheetTalk is a professional AI-powered data analysis tool.",
    },
)

# ── Professional chart color palette ────────────────────────────────
CHART_COLORS = [
    "#18181b", "#3f3f46", "#52525b", "#71717a",
    "#a1a1aa", "#d4d4d8", "#27272a", "#404040",
    "#525252", "#737373",
]

CHART_COLORS_EXTENDED = [
    "#18181b", "#3f3f46", "#52525b", "#71717a",
    "#a1a1aa", "#d4d4d8", "#e4e4e7", "#27272a",
    "#404040", "#525252",
]


def style_figure(fig):
    """Apply premium professional styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color="#3f3f46",
        ),
        title=dict(
            font=dict(size=15, color="#18181b", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left",
            pad=dict(l=4, t=4, b=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=52, r=20, t=60, b=52),
        legend=dict(
            font=dict(size=11, color="#71717a"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            xanchor="center",
            x=0.5,
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#18181b",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#fafafa",
            bordercolor="#18181b",
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#e4e4e7",
            tickfont=dict(size=11, color="#a1a1aa"),
            title_font=dict(size=12, color="#71717a"),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#f4f4f5",
            griddash="dot",
            showline=False,
            tickfont=dict(size=11, color="#a1a1aa"),
            title_font=dict(size=12, color="#71717a"),
            zeroline=False,
        ),
        bargap=0.32,
        bargroupgap=0.08,
    )

    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            trace.marker.line.width = 0
            trace.marker.cornerradius = 3
            if not trace.marker.color:
                trace.marker.color = "#18181b"
            trace.marker.opacity = 0.9

        if trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.width = 2.5
            trace.line.shape = "spline"

        if trace.type == "pie":
            trace.marker.line.width = 2.5
            trace.marker.line.color = "#ffffff"
            trace.textfont = dict(
                size=11, color="#3f3f46", family="Inter, sans-serif"
            )
            trace.marker.colors = CHART_COLORS

        if trace.type == "histogram":
            trace.marker.line.width = 1
            trace.marker.line.color = "#ffffff"
            trace.marker.opacity = 0.85

    return fig


# ── Custom Font + Premium CSS ───────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* ================================================================
   SHEETTALK — PREMIUM MONOCHROME UI
   Design: Zinc Confidence — Inspired by Linear, Vercel, Stripe
   ================================================================ */

:root {
    /* ── Zinc palette ── */
    --zinc-50:  #fafafa;
    --zinc-100: #f4f4f5;
    --zinc-200: #e4e4e7;
    --zinc-300: #d4d4d8;
    --zinc-400: #a1a1aa;
    --zinc-500: #71717a;
    --zinc-600: #52525b;
    --zinc-700: #3f3f46;
    --zinc-800: #27272a;
    --zinc-900: #18181b;
    --zinc-950: #09090b;

    /* ── Semantic mapping ── */
    --brand-primary:      var(--zinc-900);
    --brand-accent:       var(--zinc-700);
    --brand-accent-hover: var(--zinc-950);
    --brand-accent-light: var(--zinc-100);
    --brand-accent-glow:  rgba(24, 24, 27, 0.08);

    --bg-app:     #fafafa;
    --bg-surface: #ffffff;
    --bg-subtle:  var(--zinc-50);
    --bg-muted:   var(--zinc-100);
    --bg-inset:   #f7f7f8;

    --border-light:   var(--zinc-200);
    --border-default: #e4e4e7;
    --border-hover:   var(--zinc-300);
    --border-strong:  var(--zinc-400);

    --text-primary:   var(--zinc-900);
    --text-secondary: var(--zinc-500);
    --text-tertiary:  var(--zinc-400);
    --text-muted:     var(--zinc-300);
    --text-inverse:   #ffffff;

    --success:    #059669;
    --success-bg: #ecfdf5;
    --success-border: rgba(5,150,105,0.15);
    --warning:    #d97706;
    --warning-bg: #fffbeb;
    --error:      #dc2626;
    --error-bg:   #fef2f2;

    --radius-xs:   4px;
    --radius-sm:   6px;
    --radius-md:   8px;
    --radius-lg:   12px;
    --radius-xl:   16px;
    --radius-2xl:  20px;
    --radius-full: 9999px;

    --shadow-xs:  0 1px 2px rgba(0,0,0,0.03);
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
    --shadow-md:  0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -2px rgba(0,0,0,0.02);
    --shadow-lg:  0 10px 15px -3px rgba(0,0,0,0.04), 0 4px 6px -4px rgba(0,0,0,0.01);
    --shadow-xl:  0 20px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.02);
    --shadow-ring: 0 0 0 1px rgba(0,0,0,0.03);

    --transition-fast:   120ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base:   200ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── 1. GLOBAL ────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background-color: var(--bg-app) !important;
}

.main .block-container {
    max-width: 880px !important;
    padding: 1.25rem 2rem 5rem !important;
}

header[data-testid="stHeader"] {
    background: rgba(250,250,250,0.82) !important;
    backdrop-filter: saturate(180%) blur(20px) !important;
    -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--zinc-300); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--zinc-400); }

/* Selection */
::selection {
    background: rgba(24,24,27,0.1);
    color: var(--zinc-900);
}

/* ── 2. TYPOGRAPHY ────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.15 !important;
    margin-top: 0 !important;
}

h1, .stMarkdown h1 {
    font-size: 1.75rem !important;
    margin-bottom: 0.25rem !important;
    font-weight: 800 !important;
}
h2, .stMarkdown h2 {
    font-size: 1.3125rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
}
h3, .stMarkdown h3 {
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

p, .stMarkdown p, .stText {
    font-size: 0.9375rem !important;
    line-height: 1.65 !important;
    color: var(--text-secondary) !important;
}

a {
    color: var(--text-primary) !important;
    text-decoration: none !important;
    transition: opacity var(--transition-fast) !important;
    border-bottom: 1px solid var(--border-default) !important;
}
a:hover {
    opacity: 0.7 !important;
    border-bottom-color: var(--text-primary) !important;
    text-decoration: none !important;
}

/* ── 3. SIDEBAR ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.25rem 1.375rem 1rem !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.8125rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.625rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-tertiary) !important;
    margin-top: 0.375rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 0.75rem 0 !important;
}

[data-testid="stSidebar"] .stExpander {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-subtle) !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ── 4. BUTTONS ───────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: var(--zinc-900) !important;
    color: var(--text-inverse) !important;
    border: 1px solid var(--zinc-900) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5625rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    box-shadow: var(--shadow-xs), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transition: all var(--transition-base) !important;
    line-height: 1.5 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: var(--zinc-950) !important;
    border-color: var(--zinc-950) !important;
    box-shadow: var(--shadow-md), inset 0 1px 0 rgba(255,255,255,0.06) !important;
    transform: translateY(-0.5px) !important;
}

.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(0) !important;
    box-shadow: var(--shadow-xs) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5625rem 1.25rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all var(--transition-base) !important;
    letter-spacing: -0.01em !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button:hover {
    background-color: var(--bg-subtle) !important;
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stDownloadButton > button {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-xs) !important;
}

.stDownloadButton > button:hover {
    border-color: var(--border-hover) !important;
    background-color: var(--bg-subtle) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── 5. FORM INPUTS ───────────────────────────────────────────── */
.stTextInput > div > div > input {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5625rem 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-xs) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--zinc-400) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow), var(--shadow-xs) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-tertiary) !important;
}

.stTextArea > div > div > textarea {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5625rem 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-xs) !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: var(--zinc-400) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow), var(--shadow-xs) !important;
}

.stSelectbox > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    transition: all var(--transition-base) !important;
    box-shadow: var(--shadow-xs) !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--border-hover) !important;
}

.stSelectbox > div > div[aria-expanded="true"] {
    border-color: var(--zinc-400) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow), var(--shadow-xs) !important;
}

.stTextInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiSelect > label,
.stFileUploader > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.6875rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 0.25rem !important;
}

/* ── 6. FILE UPLOADER ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--bg-subtle) !important;
    border: 1.5px dashed var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--zinc-400) !important;
    background-color: var(--bg-muted) !important;
}

[data-testid="stFileUploader"] section > button {
    background-color: var(--zinc-900) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    padding: 0.375rem 0.875rem !important;
}

[data-testid="stFileUploader"] small {
    color: var(--text-tertiary) !important;
    font-size: 0.625rem !important;
}

/* ── 7. TABS ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border-light) !important;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    padding: 0.5rem 0.875rem !important;
    border: none !important;
    background-color: transparent !important;
    transition: color var(--transition-fast) !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    border-bottom-color: var(--zinc-900) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--zinc-900) !important;
    height: 2px !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.125rem !important;
}

/* ── 8. METRICS ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.875rem 1.125rem !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stMetric"] label {
    font-size: 0.625rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--text-tertiary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.375rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
}

/* ── 9. DATAFRAMES ────────────────────────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ── 10. EXPANDER ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 0.375rem !important;
    background: var(--bg-surface) !important;
}

[data-testid="stExpander"] details {
    border: none !important;
}

[data-testid="stExpander"] summary {
    padding: 0.625rem 0.875rem !important;
    font-weight: 500 !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 0.875rem 0.875rem !important;
    border-top: 1px solid var(--border-light) !important;
}

/* ── 11. ALERTS ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-size: 0.8125rem !important;
    border-left-width: 3px !important;
}

/* ── 12. PROGRESS ─────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background-color: var(--zinc-700) !important;
    border-radius: var(--radius-full) !important;
}

.stProgress > div > div {
    background-color: var(--zinc-100) !important;
    border-radius: var(--radius-full) !important;
    height: 3px !important;
}

/* ── 13. CHAT INTERFACE ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-light) !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-xs) !important;
    padding: 0.875rem 1.125rem !important;
    margin-bottom: 0.5rem !important;
    transition: box-shadow var(--transition-fast) !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-default) !important;
    padding: 0.6875rem 1rem !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--zinc-400) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow), var(--shadow-sm) !important;
}

[data-testid="stChatInput"] button {
    background-color: var(--zinc-900) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-inverse) !important;
    transition: all var(--transition-fast) !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: var(--zinc-950) !important;
}

/* ── 14. PLOTLY CHART ─────────────────────────────────────────── */
[data-testid="stPlotlyChart"], .stPlotlyChart {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.625rem !important;
    box-shadow: var(--shadow-xs) !important;
    overflow: hidden !important;
    transition: box-shadow var(--transition-base) !important;
}

[data-testid="stPlotlyChart"]:hover, .stPlotlyChart:hover {
    box-shadow: var(--shadow-sm) !important;
}

/* ── 15. CODE BLOCKS ──────────────────────────────────────────── */
div[data-testid="stCodeBlock"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-xs) !important;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', monospace !important;
    font-size: 0.8125rem !important;
}

/* ── 16. DIVIDER ──────────────────────────────────────────────── */
hr, .stMarkdown hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 1.25rem 0 !important;
}

/* ── 17. CUSTOM COMPONENTS ────────────────────────────────────── */

/* Hero */
.hero-section {
    text-align: center;
    padding: 3rem 0 1.75rem;
    max-width: 540px;
    margin: 0 auto;
}

.hero-section.animate-in {
    animation: heroFadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3125rem;
    padding: 0.25rem 0.75rem;
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-size: 0.625rem;
    font-weight: 500;
    border-radius: var(--radius-full);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-default);
    box-shadow: var(--shadow-xs);
}

.hero-badge-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--zinc-400);
    display: inline-block;
}

.hero-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 2.375rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.045em !important;
    line-height: 1.08 !important;
    margin-bottom: 0.875rem !important;
    margin-top: 0 !important;
}

.hero-subtitle {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.55 !important;
    margin-bottom: 0 !important;
    max-width: 420px;
    margin-left: auto;
    margin-right: auto;
    font-weight: 400;
}

.trust-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.25rem;
    margin-top: 1.375rem;
    padding-top: 1.125rem;
    border-top: 1px solid var(--border-light);
}

.trust-item {
    display: flex;
    align-items: center;
    gap: 0.3125rem;
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    font-weight: 400;
    letter-spacing: 0.005em;
}

.trust-item-icon {
    font-size: 0.75rem;
    opacity: 0.7;
}

/* Sidebar components */
.sidebar-brand {
    padding: 0.125rem 0 0.375rem;
}

.sidebar-brand-inner {
    display: flex;
    align-items: center;
    gap: 0.625rem;
}

.sidebar-brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: var(--radius-sm);
    background: var(--zinc-900);
    color: white;
    font-size: 0.875rem;
    flex-shrink: 0;
}

.sidebar-brand-text {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.sidebar-brand-name {
    font-size: 0.9375rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.sidebar-brand-tag {
    font-size: 0.5625rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-tertiary);
    margin-top: 1px;
}

.sidebar-label {
    font-size: 0.5625rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-tertiary) !important;
    margin-bottom: 0.375rem !important;
    margin-top: 0.125rem !important;
}

.sidebar-sep {
    height: 1px;
    background: var(--border-light);
    margin: 0.75rem 0;
}

.sidebar-meta {
    padding-top: 0.5rem;
    font-size: 0.5625rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.02em;
}

.sidebar-meta a {
    color: var(--text-tertiary) !important;
    font-weight: 500;
    border-bottom: none !important;
}

.sidebar-meta a:hover {
    color: var(--text-primary) !important;
}

/* Status indicators */
.indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.3125rem;
    padding: 0.1875rem 0.625rem;
    border-radius: var(--radius-full);
    font-size: 0.5625rem;
    font-weight: 500;
    letter-spacing: 0.015em;
}

.indicator-active {
    background: var(--success-bg);
    color: #065f46;
    border: 1px solid var(--success-border);
}

.indicator-idle {
    background: var(--bg-muted);
    color: var(--text-tertiary);
    border: 1px solid var(--border-light);
}

.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.1875rem 0.5rem;
    background: var(--bg-muted);
    color: var(--text-secondary);
    font-size: 0.5625rem;
    font-weight: 500;
    border-radius: var(--radius-full);
    border: 1px solid var(--border-light);
    margin-top: 0.25rem;
}

/* Onboarding cards */
.ob-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-xl);
    padding: 2.75rem 2.25rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    max-width: 460px;
    margin: 0 auto;
    animation: cardFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}

.ob-icon {
    width: 48px;
    height: 48px;
    border-radius: var(--radius-lg);
    background: var(--bg-muted);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    margin: 0 auto 1.375rem;
    border: 1px solid var(--border-light);
}

.ob-heading {
    font-family: 'Inter', sans-serif;
    font-size: 1.1875rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 0.4375rem;
    line-height: 1.2;
}

.ob-body {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 1.625rem;
    max-width: 360px;
    margin-left: auto;
    margin-right: auto;
}

.ob-steps {
    text-align: left;
    background: var(--bg-subtle);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    margin-bottom: 0;
}

.ob-steps p {
    margin: 0 !important;
    font-size: 0.8125rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.9 !important;
    display: flex !important;
    align-items: baseline !important;
    gap: 0.5rem !important;
}

.ob-steps strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.ob-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: var(--radius-full);
    background: var(--zinc-100);
    border: 1px solid var(--border-default);
    color: var(--text-secondary);
    font-size: 0.625rem;
    font-weight: 600;
    flex-shrink: 0;
}

.ob-examples-label {
    font-size: 0.5625rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-tertiary);
    text-align: center;
    margin-bottom: 0.625rem;
}

.ob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.375rem;
}

.ob-chip {
    display: flex;
    align-items: center;
    gap: 0.3125rem;
    padding: 0.4375rem 0.625rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-sm);
    font-size: 0.6875rem;
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    cursor: default;
    font-weight: 400;
}

.ob-chip:hover {
    border-color: var(--border-hover);
    background: var(--bg-muted);
    color: var(--text-primary);
}

.ob-chip-icon {
    font-size: 0.75rem;
    flex-shrink: 0;
    opacity: 0.65;
}

/* Ready banner */
.ready-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.375rem;
    padding: 0.5rem 0.875rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    margin-bottom: 0.75rem;
    font-size: 0.6875rem;
    color: var(--text-secondary);
    font-weight: 400;
    box-shadow: var(--shadow-xs);
    animation: cardFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.ready-bar strong {
    color: var(--text-primary);
    font-weight: 600;
}

.ready-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
    flex-shrink: 0;
}

/* Footer */
.site-footer {
    text-align: center;
    padding: 2.25rem 0 0.75rem;
    margin-top: 3.5rem;
    border-top: 1px solid var(--border-light);
}

.site-footer-name {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.site-footer-desc {
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    margin: 0.125rem 0 0.5rem;
    font-weight: 400;
}

.site-footer-links {
    font-size: 0.625rem;
    margin-bottom: 0.625rem;
}

.site-footer-links a {
    color: var(--text-tertiary) !important;
    border-bottom: none !important;
    font-weight: 400;
    transition: color var(--transition-fast) !important;
}

.site-footer-links a:hover {
    color: var(--text-primary) !important;
    opacity: 1 !important;
}

.site-footer-sep {
    color: var(--zinc-200);
    margin: 0 0.3125rem;
}

.site-footer-legal {
    font-size: 0.5625rem;
    color: var(--text-muted);
    letter-spacing: 0.015em;
}

/* ── 18. ANIMATIONS ───────────────────────────────────────────── */
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(8px) scale(0.99); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── 19. RESPONSIVE ───────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-title  { font-size: 1.75rem !important; }
    .hero-subtitle { font-size: 0.875rem !important; }
    .hero-section { padding: 2rem 0 1.25rem; }
    .ob-grid { grid-template-columns: 1fr; }
    .ob-card { padding: 2rem 1.5rem; }
    .trust-row { flex-direction: column; gap: 0.5rem; }
}

@media (max-width: 480px) {
    .hero-title { font-size: 1.5rem !important; }
    h1, .stMarkdown h1 { font-size: 1.375rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ───────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-inner">
                <div class="sidebar-brand-icon">📊</div>
                <div class="sidebar-brand-text">
                    <span class="sidebar-brand-name">SheetTalk</span>
                    <span class="sidebar-brand-tag">AI Data Analyst</span>
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sidebar-label">Connection</p>', unsafe_allow_html=True
    )

    api_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        placeholder="nvapi-…",
        label_visibility="collapsed",
    )

    model_choice = st.selectbox(
        "Model",
        [
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "meta/llama3-70b-instruct",
            "nvidia/nemotron-4-340b-instruct",
        ],
        label_visibility="collapsed",
    )

    if api_key:
        st.session_state.api_key_set = True
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        st.session_state.client = client
        st.markdown(
            '<div class="indicator indicator-active">● Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state.api_key_set = False
        st.markdown(
            '<div class="indicator indicator-idle">○ Awaiting key</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="sidebar-label">Data</p>', unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="csv_upload",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if (
            st.session_state.df is None
            or st.session_state.get("file_name") != uploaded_file.name
        ):
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []

        st.markdown(
            f'<div class="file-pill">📄 {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Preview data", expanded=False):
            st.dataframe(
                st.session_state.df.head(10), use_container_width=True
            )
            st.caption(
                f"**{st.session_state.df.shape[0]:,}** rows × "
                f"**{st.session_state.df.shape[1]}** columns"
            )

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-meta">
            v1.0 · <a href="#">Docs</a>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ── Hero ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-section animate-in">
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            AI-Powered Analysis
        </div>
        <h1 class="hero-title">Chat with your data.</h1>
        <p class="hero-subtitle">
            Upload a CSV, ask questions in plain English, and get
            instant insights with visualizations.
        </p>
        <div class="trust-row">
            <span class="trust-item">
                <span class="trust-item-icon">🔒</span>
                Session-only processing
            </span>
            <span class="trust-item">
                <span class="trust-item-icon">⚡</span>
                Real-time analysis
            </span>
            <span class="trust-item">
                <span class="trust-item-icon">📊</span>
                Auto-visualizations
            </span>
        </div>
    </div>
""",
    unsafe_allow_html=True,
)


# ── Helpers ─────────────────────────────────────────────────────────
def extract_code(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response")

    code = raw.strip()

    if "```" in code:
        parts = code.split("```")
        code_blocks = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lines = part.split("\n")
                if lines and lines[0].strip().lower() in (
                    "python",
                    "py",
                    "python3",
                    "",
                ):
                    lines = lines[1:]
                code_blocks.append("\n".join(lines))
        if code_blocks:
            code = "\n".join(code_blocks).strip()

    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        lines.append(line)

    if not lines:
        raise ValueError("LLM returned no executable code")

    return "\n".join(lines)


SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
    "round": round, "sorted": sorted, "any": any, "all": all,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "callable": callable, "iter": iter, "next": next,
    "reversed": reversed, "repr": repr,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "int": int, "float": float, "str": str, "bool": bool,
    "range": range, "slice": slice, "object": object,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "print": print, "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "KeyError": KeyError,
    "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "NameError": NameError,
}


def safe_execute(code: str, df: pd.DataFrame) -> dict:
    global_ns = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
    }
    local_ns: dict = {}

    try:
        result = eval(code, global_ns)
        return {"result": result, "fig": None}
    except SyntaxError:
        pass

    exec(code, global_ns, local_ns)

    result = local_ns.get("result", None)
    if result is None:
        non_fig = {k: v for k, v in local_ns.items() if k != "fig"}
        if non_fig:
            result = list(non_fig.values())[-1]

    fig = local_ns.get("fig", None)
    if fig is None and "fig" in global_ns and global_ns["fig"] is not None:
        fig = global_ns["fig"]

    return {"result": result, "fig": fig}


def build_system_prompt() -> str:
    return """You are a world-class Python data analyst, visualization expert, and product-grade AI assistant.

You are working inside a Streamlit app where:
- The dataset is already loaded as a pandas DataFrame named `df`
- Your job is to analyze data and generate insights

━━━━━━━━━━━━━━━━━━━━━━━
🎯 YOUR OBJECTIVE
━━━━━━━━━━━━━━━━━━━━━━━
1. Understand the user's question
2. Perform accurate data analysis using pandas/numpy
3. Return clean, meaningful results
4. Create a beautiful Plotly visualization when useful

━━━━━━━━━━━━━━━━━━━━━━━
⚠️ STRICT OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY executable Python code
- NO explanations, NO markdown, NO comments
- Do NOT include any import statements (all modules are pre-loaded)
- ALWAYS use the variable `df`
- Final output MUST be stored in: result
- If visualization is useful, ALSO create: fig

━━━━━━━━━━━━━━━━━━━━━━━
📦 PRE-LOADED MODULES
━━━━━━━━━━━━━━━━━━━━━━━
- pandas as pd
- numpy as np
- plotly.express as px
- plotly.graph_objects as go

Do NOT import anything. Just use them directly.

━━━━━━━━━━━━━━━━━━━━━━━
📊 VISUALIZATION RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Use plotly.express as px (already imported)
- Charts must look clean, modern, and professional
- Use template='plotly_white' always
- Use a monochromatic grey palette: ['#18181b', '#3f3f46', '#52525b', '#71717a', '#a1a1aa', '#d4d4d8']

AUTO-SELECT chart type:
- Comparison → px.bar
- Trends/time → px.line
- Distribution → px.histogram
- Proportions → px.pie

RULES:
- Always sort values before plotting
- Limit to top 10 categories if too many values
- Add proper title using title='...'
- Use clear axis labels
- Avoid clutter
- For bar charts, use color_discrete_sequence=['#18181b']
- For line charts, use color_discrete_sequence=['#18181b']
- For pie charts, use color_discrete_sequence=['#18181b', '#3f3f46', '#52525b', '#71717a', '#a1a1aa', '#d4d4d8']

━━━━━━━━━━━━━━━━━━━━━━━
🧠 DATA HANDLING RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Handle missing values if needed
- Use groupby for aggregations
- Use meaningful column names
- Round numbers when necessary
- Return clean outputs (not raw messy data)

━━━━━━━━━━━━━━━━━━━━━━━
💬 OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━
- The result should feel clean and readable
- Avoid returning huge raw tables
- Prefer summaries, grouped data, or key insights
- Always assign final answer to `result`
- Always assign plotly figure to `fig` (when a chart is created)"""


def build_user_prompt(question: str, df: pd.DataFrame) -> str:
    dtypes = df.dtypes.to_string()
    sample = df.head(5).to_string()
    shape = f"{df.shape[0]} rows × {df.shape[1]} columns"

    return f"""
📂 DATA CONTEXT
Shape: {shape}

Columns and types:
{dtypes}

Sample rows:
{sample}

━━━━━━━━━━━━━━━━━━━━━━━
Question: {question}
"""


# ── Chat loop ──────────────────────────────────────────────────────
if st.session_state.api_key_set and st.session_state.df is not None:

    df = st.session_state.df

    st.markdown(
        f"""
        <div class="ready-bar">
            <span class="ready-dot"></span>
            Ready — analyzing <strong>{st.session_state.get("file_name", "your file")}</strong>
            &nbsp;·&nbsp; {df.shape[0]:,} rows × {df.shape[1]} columns
        </div>
    """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            if "code" in msg:
                with st.expander("View generated code", expanded=False):
                    st.code(msg["code"], language="python")
            if "result" in msg and msg["result"] is not None:
                result_val = msg["result"]
                if isinstance(result_val, pd.DataFrame):
                    st.dataframe(result_val, use_container_width=True)
                elif isinstance(result_val, pd.Series):
                    st.dataframe(
                        result_val.to_frame(), use_container_width=True
                    )
                else:
                    st.write(result_val)
            if "fig" in msg and msg["fig"] is not None:
                styled_fig = style_figure(msg["fig"])
                st.plotly_chart(styled_fig, use_container_width=True)
            if "chart" in msg and msg["chart"] is not None:
                try:
                    st.bar_chart(msg["chart"])
                except Exception:
                    pass

    if prompt := st.chat_input("Ask anything about your data…"):

        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        try:
            client = st.session_state.client

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {
                        "role": "user",
                        "content": build_user_prompt(prompt, df),
                    },
                ],
                temperature=0,
                max_tokens=1024,
            )

            raw = response.choices[0].message.content
            code = extract_code(raw)
            output = safe_execute(code, df)

            result = output["result"]
            fig = output["fig"]

            assistant_msg: dict = {"role": "assistant", "code": code}

            if result is not None:
                assistant_msg["result"] = result

            if fig is not None:
                assistant_msg["fig"] = fig

            if fig is None:
                if (
                    isinstance(result, pd.Series)
                    and result.dtype.kind in "iufb"
                ):
                    assistant_msg["chart"] = result
                elif isinstance(result, pd.DataFrame):
                    num = result.select_dtypes(include="number")
                    if not num.empty and len(num) <= 50:
                        assistant_msg["chart"] = num

            st.session_state.chat_history.append(assistant_msg)

        except Exception as e:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Something went wrong: {str(e)}",
                }
            )

        st.rerun()

# ── Fallback: No API key ──────────────────────────────────────────
elif not st.session_state.api_key_set:
    st.markdown(
        """
        <div class="ob-card">
            <div class="ob-icon">🔑</div>
            <div class="ob-heading">Connect your NVIDIA API</div>
            <div class="ob-body">
                Enter your API key in the sidebar to unlock AI-powered
                data analysis and visualization.
            </div>
            <div class="ob-steps">
                <p><span class="ob-step-num">1</span> Visit <a href="https://build.nvidia.com/" target="_blank">build.nvidia.com</a></p>
                <p><span class="ob-step-num">2</span> Create an account or sign in</p>
                <p><span class="ob-step-num">3</span> Generate an API key</p>
                <p><span class="ob-step-num">4</span> Paste it in the sidebar</p>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )

# ── Fallback: No data ────────────────────────────────────────────
elif st.session_state.df is None:
    st.markdown(
        """
        <div class="ob-card">
            <div class="ob-icon">📂</div>
            <div class="ob-heading">Upload your dataset</div>
            <div class="ob-body">
                Drop a CSV file in the sidebar to start exploring
                your data with natural language.
            </div>
            <div class="ob-examples-label">Example questions</div>
            <div class="ob-grid">
                <div class="ob-chip">
                    <span class="ob-chip-icon">📊</span>
                    Sales by region
                </div>
                <div class="ob-chip">
                    <span class="ob-chip-icon">📈</span>
                    Trend over time
                </div>
                <div class="ob-chip">
                    <span class="ob-chip-icon">🔢</span>
                    Average revenue
                </div>
                <div class="ob-chip">
                    <span class="ob-chip-icon">🏆</span>
                    Top 10 customers
                </div>
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# ── Footer ──────────────────────────────────────────────────────────
year = datetime.now().year
st.markdown(
    f"""
    <div class="site-footer">
        <div class="site-footer-name">SheetTalk</div>
        <div class="site-footer-desc">AI-powered data analysis for professionals.</div>
        <div class="site-footer-links">
            <a href="#">Privacy</a>
            <span class="site-footer-sep">·</span>
            <a href="#">Terms</a>
            <span class="site-footer-sep">·</span>
            <a href="#">GitHub</a>
        </div>
        <div class="site-footer-legal">
            © {year} SheetTalk · Made with ❤️ by Dhokla
        </div>
    </div>
""",
    unsafe_allow_html=True,
)