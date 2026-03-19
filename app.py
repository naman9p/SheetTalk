import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from openai import OpenAI

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
    "#374151", "#6b7280", "#9ca3af", "#4b5563",
    "#1f2937", "#d1d5db", "#111827", "#e5e7eb",
    "#525252", "#a3a3a3",
]

CHART_COLORS_EXTENDED = [
    "#374151", "#6b7280", "#9ca3af", "#4b5563",
    "#1f2937", "#d1d5db", "#78716c", "#a8a29e",
    "#57534e", "#44403c",
]


def style_figure(fig):
    """Apply premium professional styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color="#374151",
        ),
        title=dict(
            font=dict(size=16, color="#111827", family="Inter, sans-serif"),
            x=0,
            xanchor="left",
            pad=dict(l=8, t=8),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=56, b=48),
        legend=dict(
            font=dict(size=11, color="#6b7280"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#ffffff",
            bordercolor="#1f2937",
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#e5e7eb",
            tickfont=dict(size=11, color="#9ca3af"),
            title_font=dict(size=12, color="#6b7280"),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#f3f4f6",
            griddash="dot",
            showline=False,
            tickfont=dict(size=11, color="#9ca3af"),
            title_font=dict(size=12, color="#6b7280"),
            zeroline=False,
        ),
        bargap=0.3,
        bargroupgap=0.1,
    )

    # Style bars
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            trace.marker.line.width = 0
            trace.marker.cornerradius = 4
            if not trace.marker.color:
                trace.marker.color = "#374151"
            trace.marker.opacity = 0.92

        # Style lines
        if trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.width = 2.5
            trace.line.shape = "spline"

        # Style pie
        if trace.type == "pie":
            trace.marker.line.width = 2
            trace.marker.line.color = "#ffffff"
            trace.textfont = dict(size=11, color="#374151", family="Inter, sans-serif")
            trace.marker.colors = CHART_COLORS

        # Style histogram
        if trace.type == "histogram":
            trace.marker.line.width = 1
            trace.marker.line.color = "#ffffff"
            trace.marker.opacity = 0.88

    return fig


# ── Custom Font + Premium CSS ───────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ================================================================
   SHEETTALK — GREY PREMIUM UI
   Design System: Monochrome Confidence
   ================================================================ */

:root {
    --brand-primary: #111827;
    --brand-accent: #374151;
    --brand-accent-hover: #1f2937;
    --brand-accent-light: #f3f4f6;
    --brand-accent-glow: rgba(55, 65, 81, 0.12);
    --brand-accent-mid: #6b7280;

    --bg-primary: #fafafa;
    --bg-surface: #ffffff;
    --bg-subtle: #f9fafb;
    --bg-muted: #f3f4f6;

    --border-default: #e5e7eb;
    --border-hover: #d1d5db;
    --border-strong: #9ca3af;

    --text-primary: #111827;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --text-inverse: #ffffff;

    --color-success: #059669;
    --color-success-bg: #ecfdf5;
    --color-success-border: rgba(5, 150, 105, 0.2);
    --color-warning: #d97706;
    --color-warning-bg: #fffbeb;
    --color-error: #dc2626;
    --color-error-bg: #fef2f2;
    --color-info: #4b5563;
    --color-info-bg: #f9fafb;

    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-full: 9999px;

    --shadow-xs: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.03);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -4px rgba(0,0,0,0.02);

    --transition-fast: 150ms ease;
    --transition-base: 200ms ease;
}

/* ── 1. GLOBAL ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--bg-primary) !important;
}

.main .block-container {
    max-width: 940px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

header[data-testid="stHeader"] {
    background: rgba(250, 250, 250, 0.88) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid var(--border-default) !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }

/* ── 2. TYPOGRAPHY ────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin-top: 0 !important;
}

h1, .stMarkdown h1 { font-size: 1.875rem !important; margin-bottom: 0.25rem !important; }
h2, .stMarkdown h2 { font-size: 1.375rem !important; font-weight: 600 !important; }
h3, .stMarkdown h3 { font-size: 1.125rem !important; font-weight: 600 !important; }

p, .stMarkdown p, .stText {
    font-size: 0.9375rem !important;
    line-height: 1.65 !important;
    color: var(--text-secondary) !important;
}

a { color: var(--brand-accent) !important; text-decoration: none !important; transition: color var(--transition-fast) !important; }
a:hover { color: var(--brand-primary) !important; text-decoration: underline !important; }

/* ── 3. SIDEBAR ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #f8f9fa !important;
    border-right: 1px solid var(--border-default) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.8125rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: var(--text-tertiary) !important;
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid var(--border-default) !important;
    margin: 0.875rem 0 !important;
}

[data-testid="stSidebar"] .stExpander {
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
}

/* ── 4. BUTTONS ───────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: var(--brand-primary) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.625rem 1.75rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08), 0 1px 1px rgba(0,0,0,0.04) !important;
    transition: all var(--transition-base) !important;
    line-height: 1.5 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #000000 !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.12), 0 2px 4px rgba(0,0,0,0.06) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button {
    background-color: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.625rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all var(--transition-base) !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button:hover {
    border-color: var(--border-hover) !important;
    background-color: var(--bg-subtle) !important;
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
}

.stDownloadButton > button:hover {
    border-color: var(--brand-accent) !important;
    background-color: var(--brand-accent-light) !important;
}

/* ── 5. FORM INPUTS ───────────────────────────────────────────── */
.stTextInput > div > div > input {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.625rem 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--brand-accent) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder { color: var(--text-tertiary) !important; }

.stTextArea > div > div > textarea {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.625rem 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: var(--brand-accent) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow) !important;
}

.stSelectbox > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    transition: all var(--transition-base) !important;
}

.stSelectbox > div > div:hover { border-color: var(--border-hover) !important; }

.stSelectbox > div > div[aria-expanded="true"] {
    border-color: var(--brand-accent) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow) !important;
}

.stTextInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiSelect > label,
.stFileUploader > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 0.25rem !important;
}

/* ── 6. FILE UPLOADER ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--bg-subtle) !important;
    border: 1.5px dashed var(--border-hover) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.25rem !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--brand-accent) !important;
    background-color: var(--brand-accent-light) !important;
}

[data-testid="stFileUploader"] section > button {
    background-color: var(--brand-primary) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
}

[data-testid="stFileUploader"] small {
    color: var(--text-tertiary) !important;
    font-size: 0.6875rem !important;
}

/* ── 7. TABS ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border-default) !important;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    padding: 0.625rem 1rem !important;
    border: none !important;
    background-color: transparent !important;
    transition: all var(--transition-fast) !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
}

.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; }

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--brand-primary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--brand-primary) !important;
    height: 2px !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── 8. METRICS ───────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-sm) !important;
    border-color: var(--border-hover) !important;
}

[data-testid="stMetric"] label {
    font-size: 0.6875rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-tertiary) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

/* ── 9. DATAFRAMES ────────────────────────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ── 10. EXPANDER ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stExpander"] details { border: none !important; }

[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem !important;
    border-top: 1px solid var(--border-default) !important;
}

/* ── 11. ALERTS ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-size: 0.8125rem !important;
    border-left: 3px solid var(--border-strong) !important;
}

/* ── 12. PROGRESS & SPINNER ───────────────────────────────────── */
.stProgress > div > div > div > div {
    background-color: var(--brand-accent) !important;
    border-radius: var(--radius-full) !important;
}

.stProgress > div > div {
    background-color: var(--bg-muted) !important;
    border-radius: var(--radius-full) !important;
    height: 4px !important;
}

/* ── 13. CHAT INTERFACE ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-default) !important;
    background: var(--bg-surface) !important;
    box-shadow: var(--shadow-xs) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.625rem !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9375rem !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-default) !important;
    padding: 0.75rem 1rem !important;
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--brand-accent) !important;
    box-shadow: 0 0 0 3px var(--brand-accent-glow) !important;
}

[data-testid="stChatInput"] button {
    background-color: var(--brand-primary) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-inverse) !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #000000 !important;
}

/* ── 14. PLOTLY CHART ─────────────────────────────────────────── */
[data-testid="stPlotlyChart"], .stPlotlyChart {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.75rem !important;
    box-shadow: var(--shadow-xs) !important;
    overflow: hidden !important;
}

/* ── 15. CODE BLOCKS ──────────────────────────────────────────── */
div[data-testid="stCodeBlock"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-default) !important;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace !important;
    font-size: 0.8125rem !important;
}

/* ── 16. DIVIDER ──────────────────────────────────────────────── */
hr, .stMarkdown hr {
    border: none !important;
    border-top: 1px solid var(--border-default) !important;
    margin: 1.5rem 0 !important;
}

/* ── 17. CUSTOM COMPONENTS ────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    max-width: 580px;
    margin: 0 auto;
    animation: fadeIn 0.5s ease-out;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.3rem 0.875rem;
    background: var(--brand-accent-light);
    color: var(--brand-accent);
    font-size: 0.6875rem;
    font-weight: 600;
    border-radius: var(--radius-full);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    border: 1px solid var(--border-default);
}

.hero-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.035em !important;
    line-height: 1.12 !important;
    margin-bottom: 0.75rem !important;
    margin-top: 0 !important;
}

.hero-subtitle {
    font-size: 1rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.6 !important;
    margin-bottom: 0 !important;
    max-width: 460px;
    margin-left: auto;
    margin-right: auto;
}

.trust-signal {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.625rem 0;
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    margin-top: 0.75rem;
    letter-spacing: 0.01em;
}

.sidebar-logo {
    padding: 0.25rem 0 0.5rem;
    margin-bottom: 0.25rem;
}

.logo-mark {
    display: flex;
    align-items: center;
    gap: 0.625rem;
}

.logo-icon-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: var(--radius-md);
    background: var(--brand-primary);
    font-size: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

.logo-text-group {
    display: flex;
    flex-direction: column;
}

.logo-text {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    line-height: 1.2;
}

.logo-tag {
    font-size: 0.5625rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-tertiary);
    margin-top: 0.0625rem;
}

.sidebar-section-label {
    font-size: 0.625rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-tertiary) !important;
    margin-bottom: 0.5rem !important;
    margin-top: 0.25rem !important;
    padding: 0 !important;
}

.sidebar-divider {
    height: 1px;
    background: var(--border-default);
    margin: 0.875rem 0;
}

.sidebar-footer {
    padding-top: 0.75rem;
    font-size: 0.625rem;
    color: var(--text-tertiary);
    text-align: center;
    letter-spacing: 0.01em;
}

.sidebar-footer a { color: var(--text-tertiary) !important; font-weight: 500; }
.sidebar-footer a:hover { color: var(--text-primary) !important; }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.3rem 0.75rem;
    border-radius: var(--radius-full);
    font-size: 0.625rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

.status-connected {
    background: var(--color-success-bg);
    color: #065f46;
    border: 1px solid var(--color-success-border);
}

.status-waiting {
    background: var(--bg-muted);
    color: var(--text-tertiary);
    border: 1px solid var(--border-default);
}

.data-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.625rem;
    background: var(--brand-accent-light);
    color: var(--brand-accent);
    font-size: 0.625rem;
    font-weight: 600;
    border-radius: var(--radius-full);
    letter-spacing: 0.02em;
    border: 1px solid var(--border-default);
}

.onboarding-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-xl);
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    max-width: 500px;
    margin: 0 auto;
    animation: fadeIn 0.5s ease-out;
}

.onboarding-icon {
    font-size: 2rem;
    margin-bottom: 1.25rem;
    display: block;
    opacity: 0.8;
}

.onboarding-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    margin-bottom: 0.5rem;
}

.onboarding-description {
    font-size: 0.875rem;
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 1.5rem;
}

.onboarding-steps {
    text-align: left;
    background: var(--bg-subtle);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}

.onboarding-steps p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.8125rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.85 !important;
}

.onboarding-steps strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.examples-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.example-chip {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    font-size: 0.6875rem;
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    cursor: default;
}

.example-chip:hover {
    border-color: var(--brand-accent);
    background: var(--brand-accent-light);
    color: var(--text-primary);
}

.example-icon {
    font-size: 0.8125rem;
    flex-shrink: 0;
}

.chat-ready-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.625rem 1rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    margin-bottom: 1rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
    animation: fadeIn 0.4s ease-out;
}

.chat-ready-banner strong {
    color: var(--text-primary);
}

.app-footer {
    text-align: center;
    padding: 2rem 0 0.5rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border-default);
}

.footer-brand {
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.125rem;
}

.footer-tagline {
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    margin-bottom: 0.625rem;
}

.footer-links { font-size: 0.6875rem; margin-bottom: 0.5rem; }
.footer-links a { color: var(--text-secondary) !important; text-decoration: none !important; }
.footer-links a:hover { color: var(--text-primary) !important; }

.footer-dot {
    color: var(--border-default);
    margin: 0 0.375rem;
}

.footer-copyright {
    font-size: 0.5625rem;
    color: var(--text-tertiary);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* ── 18. ANIMATIONS ───────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── 19. RESPONSIVE ───────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-title { font-size: 1.625rem !important; }
    .hero-subtitle { font-size: 0.875rem !important; }
    .hero-section { padding: 1.5rem 0 1rem; }
    .examples-grid { grid-template-columns: 1fr; }
    .onboarding-card { padding: 1.75rem 1.25rem; }
}

@media (max-width: 480px) {
    .hero-title { font-size: 1.375rem !important; }
    h1, .stMarkdown h1 { font-size: 1.375rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-mark">
                <div class="logo-icon-wrapper">📊</div>
                <div class="logo-text-group">
                    <span class="logo-text">SheetTalk</span>
                    <span class="logo-tag">AI Data Analyst</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Connection</p>', unsafe_allow_html=True)

    api_key = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-…", label_visibility="collapsed")

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
        st.markdown('<div class="status-badge status-connected">● Connected</div>', unsafe_allow_html=True)
    else:
        st.session_state.api_key_set = False
        st.markdown('<div class="status-badge status-waiting">○ Awaiting key</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Data Source</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload", label_visibility="collapsed")

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
            f'<div class="data-badge">📄 {uploaded_file.name}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Preview data", expanded=False):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            st.caption(
                f"**{st.session_state.df.shape[0]}** rows × "
                f"**{st.session_state.df.shape[1]}** columns"
            )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
        <div class="sidebar-footer">
            <span>v1.0</span> · <a href="#">Docs</a>
        </div>
    """, unsafe_allow_html=True)


# ── Main ────────────────────────────────────────────────────────────

st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">◆ AI-Powered Analysis</div>
        <h1 class="hero-title">Chat with your data.</h1>
        <p class="hero-subtitle">
            Upload a CSV, ask questions in plain English, and get
            instant insights with visualizations.
        </p>
        <div class="trust-signal">
            🔒 Data stays in your session · Never stored
        </div>
    </div>
""", unsafe_allow_html=True)


# ── Helper: extract executable code ────────────────────────────────
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
                    "python", "py", "python3", ""
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


# ── Safe builtins whitelist ─────────────────────────────────────────
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


# ── Build prompts ──────────────────────────────────────────────────
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
- Use a monochromatic grey palette: ['#374151', '#6b7280', '#9ca3af', '#4b5563', '#1f2937', '#d1d5db']

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
- For bar charts, use color_discrete_sequence=['#374151']
- For line charts, use color_discrete_sequence=['#374151']
- For pie charts, use color_discrete_sequence=['#374151', '#6b7280', '#9ca3af', '#4b5563', '#1f2937', '#d1d5db']

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

    st.markdown(f"""
        <div class="chat-ready-banner">
            ✓ Ready — analyzing <strong>{st.session_state.get("file_name", "your file")}</strong>
            &nbsp;·&nbsp; {df.shape[0]:,} rows × {df.shape[1]} columns
        </div>
    """, unsafe_allow_html=True)

    # ── Show chat history ──
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

    # ── User input ──
    if prompt := st.chat_input("Ask anything about your data…"):

        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        try:
            client = st.session_state.client

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {
                        "role": "system",
                        "content": build_system_prompt(),
                    },
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

            assistant_msg: dict = {
                "role": "assistant",
                "code": code,
            }

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
    st.markdown("""
        <div class="onboarding-card">
            <span class="onboarding-icon">🔑</span>
            <div class="onboarding-title">Connect your NVIDIA API</div>
            <div class="onboarding-description">
                Enter your API key in the sidebar to unlock AI-powered
                data analysis and visualization.
            </div>
            <div class="onboarding-steps">
                <p><strong>1.</strong> Visit <a href="https://build.nvidia.com/" target="_blank">build.nvidia.com</a></p>
                <p><strong>2.</strong> Create an account or sign in</p>
                <p><strong>3.</strong> Generate an API key</p>
                <p><strong>4.</strong> Paste it in the sidebar</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── Fallback: No data ────────────────────────────────────────────
elif st.session_state.df is None:
    st.markdown("""
        <div class="onboarding-card">
            <span class="onboarding-icon">📂</span>
            <div class="onboarding-title">Upload your dataset</div>
            <div class="onboarding-description">
                Drop a CSV file in the sidebar to start exploring your data
                with natural language queries.
            </div>
            <div style="margin-top: 0.25rem; margin-bottom: 0.5rem;">
                <p class="sidebar-section-label" style="text-align:center; margin-bottom: 0.75rem !important;">Example questions</p>
            </div>
            <div class="examples-grid">
                <div class="example-chip">
                    <span class="example-icon">📊</span>
                    Sales by region
                </div>
                <div class="example-chip">
                    <span class="example-icon">📈</span>
                    Trend over time
                </div>
                <div class="example-chip">
                    <span class="example-icon">🔢</span>
                    Average revenue
                </div>
                <div class="example-chip">
                    <span class="example-icon">🏆</span>
                    Top 10 customers
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────
st.markdown("""
    <div class="app-footer">
        <div class="footer-brand">SheetTalk</div>
        <div class="footer-tagline">AI-powered data analysis for professionals.</div>
        <div class="footer-links">
            <a href="#">Privacy</a>
            <span class="footer-dot">·</span>
            <a href="#">Terms</a>
            <span class="footer-dot">·</span>
            <a href="#">GitHub</a>
        </div>
        <div class="footer-copyright">© 2025 SheetTalk. All rights reserved.</div>
    </div>
""", unsafe_allow_html=True)