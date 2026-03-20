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

# ── Airbnb-inspired chart color palette ────────────────────────────
CHART_COLORS = [
    "#FF385C", "#00A699", "#FC642D", "#484848",
    "#767676", "#E07912", "#D93B30", "#008489",
    "#914669", "#CF1F5B",
]

CHART_COLORS_EXTENDED = CHART_COLORS[:]


def style_figure(fig):
    """Apply Airbnb-inspired clean styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color="#484848",
        ),
        title=dict(
            font=dict(size=16, color="#222222", family="Inter, sans-serif"),
            x=0.01,
            xanchor="left",
            pad=dict(l=4, t=4, b=12),
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=52, r=24, t=64, b=52),
        legend=dict(
            font=dict(size=11, color="#484848"),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        colorway=CHART_COLORS,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#222222",
            bordercolor="#DDDDDD",
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#EBEBEB",
            tickfont=dict(size=11, color="#717171"),
            title_font=dict(size=12, color="#484848"),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(235,235,235,0.7)",
            griddash="dot",
            showline=False,
            tickfont=dict(size=11, color="#717171"),
            title_font=dict(size=12, color="#484848"),
            zeroline=False,
        ),
        bargap=0.3,
        bargroupgap=0.08,
    )

    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            trace.marker.line.width = 0
            trace.marker.cornerradius = 6
            if not trace.marker.color:
                trace.marker.color = "#FF385C"
            trace.marker.opacity = 0.92

        if trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.width = 2.5
            trace.line.shape = "spline"

        if trace.type == "pie":
            trace.marker.line.width = 2.5
            trace.marker.line.color = "#FFFFFF"
            trace.textfont = dict(
                size=11, color="#222222", family="Inter, sans-serif"
            )
            trace.marker.colors = CHART_COLORS

        if trace.type == "histogram":
            trace.marker.line.width = 1
            trace.marker.line.color = "#FFFFFF"
            trace.marker.opacity = 0.88

    return fig


# ── Google Font + Airbnb-Inspired Light CSS ──────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* ================================================================
   SHEETTALK — AIRBNB-INSPIRED LIGHT THEME
   Palette: Background #FFFFFF | Surface #F7F7F7 | Card #FFFFFF
            Text #222222 | Accent #FF385C
   ================================================================ */

:root {
    /* ── Airbnb palette ── */
    --white:      #FFFFFF;
    --gray-50:    #F7F7F7;
    --gray-100:   #EBEBEB;
    --gray-200:   #DDDDDD;
    --gray-300:   #C4C4C4;
    --gray-400:   #B0B0B0;
    --gray-500:   #717171;
    --gray-600:   #484848;
    --gray-700:   #333333;
    --gray-800:   #222222;
    --gray-900:   #121212;

    /* ── Semantic mapping ── */
    --primary:            #FF385C;
    --primary-hover:      #E31C5F;
    --primary-light:      #FFF0F3;
    --primary-dim:        rgba(255, 56, 92, 0.08);
    --primary-glow:       rgba(255, 56, 92, 0.12);
    --primary-border:     rgba(255, 56, 92, 0.2);

    --teal:               #00A699;
    --teal-light:         #E6F9F7;
    --orange:             #FC642D;

    --bg-app:             #FFFFFF;
    --bg-sidebar:         #F7F7F7;
    --bg-surface:         #FFFFFF;
    --bg-subtle:          #F7F7F7;
    --bg-muted:           #F0F0F0;
    --bg-inset:           #F7F7F7;
    --bg-input:           #FFFFFF;

    --border-light:       #EBEBEB;
    --border-default:     #DDDDDD;
    --border-hover:       #B0B0B0;
    --border-focus:       #222222;

    --text-primary:       #222222;
    --text-secondary:     #484848;
    --text-tertiary:      #717171;
    --text-muted:         #B0B0B0;
    --text-inverse:       #FFFFFF;

    --success:            #00A699;
    --success-bg:         #E6F9F7;
    --success-border:     rgba(0, 166, 153, 0.25);
    --warning:            #FC642D;
    --warning-bg:         #FFF3ED;
    --error:              #C13515;
    --error-bg:           #FFF0ED;

    --radius-xs:    4px;
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    16px;
    --radius-xl:    20px;
    --radius-2xl:   24px;
    --radius-full:  9999px;

    --shadow-xs:   0 1px 2px rgba(0,0,0,0.04);
    --shadow-sm:   0 2px 4px rgba(0,0,0,0.06);
    --shadow-md:   0 4px 12px rgba(0,0,0,0.08);
    --shadow-lg:   0 8px 24px rgba(0,0,0,0.10);
    --shadow-xl:   0 12px 36px rgba(0,0,0,0.12);
    --shadow-card: 0 6px 20px rgba(0,0,0,0.06);
    --shadow-card-hover: 0 8px 28px rgba(0,0,0,0.12);

    --transition-fast:    150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base:    200ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth:  300ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-spring:  400ms cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── 1. GLOBAL ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.stApp {
    background-color: var(--bg-app) !important;
}

.main .block-container {
    max-width: 860px !important;
    padding: 1.5rem 2rem 5rem !important;
}

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: saturate(180%) blur(20px) !important;
    -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
    border-bottom: 1px solid var(--border-light) !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--gray-300); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--gray-400); }

/* Selection */
::selection {
    background: var(--primary-dim);
    color: var(--text-primary);
}

/* ── 2. TYPOGRAPHY ─────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.18 !important;
    margin-top: 0 !important;
}

h1, .stMarkdown h1 {
    font-size: 1.75rem !important;
    margin-bottom: 0.25rem !important;
    font-weight: 800 !important;
}
h2, .stMarkdown h2 {
    font-size: 1.375rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}
h3, .stMarkdown h3 {
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}

p, .stMarkdown p, .stText {
    font-size: 0.9375rem !important;
    line-height: 1.6 !important;
    color: var(--text-secondary) !important;
}

a {
    color: var(--text-primary) !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
    transition: color var(--transition-fast) !important;
}
a:hover {
    color: var(--primary) !important;
}

/* ── 3. SIDEBAR ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.375rem 1rem !important;
    background-color: var(--bg-sidebar) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background-color: var(--bg-sidebar) !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.8125rem !important;
    color: var(--text-tertiary) !important;
}

[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.625rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
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
    background: var(--white) !important;
    box-shadow: var(--shadow-xs) !important;
}

[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] {
    background: var(--white) !important;
}

/* ── 4. BUTTONS ────────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(to right, #E61E4D, #E31C5F, #D70466) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.625rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all var(--transition-base) !important;
    line-height: 1.5 !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: scale(1.01) !important;
}

.stButton > button[kind="primary"]:active,
.stButton > button[data-testid="baseButton-primary"]:active {
    transform: scale(0.98) !important;
}

.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
.stButton > button {
    background-color: var(--white) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.625rem 1.25rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: all var(--transition-base) !important;
    letter-spacing: -0.01em !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button:hover {
    background-color: var(--bg-subtle) !important;
    border-color: var(--text-primary) !important;
    box-shadow: var(--shadow-xs) !important;
    color: var(--text-primary) !important;
}

.stButton > button:active {
    transform: scale(0.98) !important;
}

.stDownloadButton > button {
    background-color: var(--white) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    transition: all var(--transition-base) !important;
}

.stDownloadButton > button:hover {
    border-color: var(--text-primary) !important;
    background-color: var(--bg-subtle) !important;
}

/* ── 5. FORM INPUTS ────────────────────────────────────────── */

/* Text Input */
.stTextInput > div > div > input {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.625rem 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(34, 34, 34, 0.12) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

/* Text Area */
.stTextArea > div > div > textarea {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.625rem 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-base) !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(34, 34, 34, 0.12) !important;
}

.stTextArea > div > div > textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── SELECTBOX ─────────────────────────────────────────────── */
.stSelectbox > div > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    transition: all var(--transition-base) !important;
    box-shadow: none !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--border-hover) !important;
}

.stSelectbox > div > div[aria-expanded="true"] {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px rgba(34, 34, 34, 0.12) !important;
}

.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] .css-1dimb5e-singleValue,
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div,
.stSelectbox [data-baseweb="select"] > div > div > div {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}

.stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"],
.stSelectbox [data-baseweb="select"] .css-1wa3eu0-placeholder {
    color: var(--text-muted) !important;
}

.stSelectbox [data-baseweb="select"] svg {
    fill: var(--text-tertiary) !important;
    color: var(--text-tertiary) !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
    overflow: hidden !important;
}

[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {
    background-color: var(--white) !important;
    border: none !important;
}

[data-baseweb="menu"] [role="option"],
[role="listbox"] [role="option"],
[data-baseweb="menu"] li,
[role="listbox"] li {
    background-color: var(--white) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0.875rem !important;
    transition: background var(--transition-fast) !important;
}

[data-baseweb="menu"] [role="option"]:hover,
[role="listbox"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
[role="listbox"] li:hover,
[data-baseweb="menu"] [data-highlighted],
[data-baseweb="menu"] [aria-selected="true"],
[role="listbox"] [aria-selected="true"] {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}

/* MultiSelect */
.stMultiSelect [data-baseweb="select"] {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
}

.stMultiSelect [data-baseweb="select"] span,
.stMultiSelect [data-baseweb="tag"] {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-xs) !important;
}

/* Labels */
.stTextInput > label,
.stTextArea > label,
.stSelectbox > label,
.stMultiSelect > label,
.stFileUploader > label,
.stSlider > label,
.stCheckbox > label,
.stRadio > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 0.375rem !important;
}

/* ── 6. FILE UPLOADER ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: var(--bg-subtle) !important;
    border: 2px dashed var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.125rem !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary) !important;
    background-color: var(--primary-light) !important;
}

[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}

[data-testid="stFileUploader"] section > button {
    background: linear-gradient(to right, #E61E4D, #E31C5F, #D70466) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    padding: 0.375rem 1rem !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: var(--text-tertiary) !important;
    font-size: 0.6875rem !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    color: var(--text-primary) !important;
}

/* ── 7. TABS ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid var(--border-light) !important;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-tertiary) !important;
    padding: 0.625rem 1rem !important;
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
    border-bottom-color: var(--text-primary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--text-primary) !important;
    height: 2px !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem !important;
}

/* ── 8. METRICS ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--shadow-xs) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
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
    letter-spacing: -0.03em !important;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: var(--success) !important;
}

/* ── 9. DATAFRAMES ─────────────────────────────────────────── */
[data-testid="stDataFrame"],
.stDataFrame {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
.dvn-scroller {
    background-color: var(--white) !important;
}

/* ── 10. EXPANDER ──────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 0.5rem !important;
    background: var(--white) !important;
}

[data-testid="stExpander"] details {
    border: none !important;
    background: var(--white) !important;
}

[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
    background: var(--white) !important;
    transition: background var(--transition-fast) !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--bg-subtle) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem !important;
    border-top: 1px solid var(--border-light) !important;
    background: var(--white) !important;
}

[data-testid="stExpander"] summary svg {
    color: var(--text-tertiary) !important;
    fill: var(--text-tertiary) !important;
}

/* ── 11. ALERTS ────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-size: 0.8125rem !important;
    border-left-width: 3px !important;
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
}

/* ── 12. PROGRESS ──────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(to right, #E61E4D, #E31C5F, #D70466) !important;
    border-radius: var(--radius-full) !important;
}

.stProgress > div > div {
    background-color: var(--bg-subtle) !important;
    border-radius: var(--radius-full) !important;
    height: 4px !important;
}

/* ── 13. CHAT INTERFACE ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important;
    padding: 1.125rem 1.375rem !important;
    margin-bottom: 0.75rem !important;
    transition: all var(--transition-fast) !important;
    border: none !important;
    box-shadow: none !important;
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border-light) !important;
}

/* User messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--white) !important;
    border: 1px solid var(--border-light) !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 0.9375rem !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-top: 1px solid var(--border-light) !important;
    background: var(--bg-app) !important;
    padding-top: 0.875rem !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9375rem !important;
    border-radius: var(--radius-full) !important;
    border: 2px solid var(--border-default) !important;
    padding: 0.75rem 1.25rem !important;
    background: var(--white) !important;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--text-primary) !important;
    box-shadow: var(--shadow-md) !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(to right, #E61E4D, #E31C5F, #D70466) !important;
    border-radius: var(--radius-full) !important;
    color: var(--text-inverse) !important;
    transition: all var(--transition-fast) !important;
    border: none !important;
}

[data-testid="stChatInput"] button:hover {
    transform: scale(1.05) !important;
    box-shadow: var(--shadow-md) !important;
}

/* Chat avatars */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    background-color: var(--text-primary) !important;
    border: none !important;
}

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #E61E4D, #D70466) !important;
    border: none !important;
}

/* ── 14. PLOTLY CHART ──────────────────────────────────────── */
[data-testid="stPlotlyChart"],
.stPlotlyChart {
    background: var(--white) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.75rem !important;
    box-shadow: var(--shadow-card) !important;
    overflow: hidden !important;
    transition: all var(--transition-base) !important;
}

[data-testid="stPlotlyChart"]:hover,
.stPlotlyChart:hover {
    box-shadow: var(--shadow-card-hover) !important;
}

/* ── 15. CODE BLOCKS ───────────────────────────────────────── */
div[data-testid="stCodeBlock"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-light) !important;
    box-shadow: var(--shadow-xs) !important;
}

div[data-testid="stCodeBlock"] pre {
    background-color: var(--gray-800) !important;
}

code {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace !important;
    font-size: 0.8125rem !important;
    color: var(--text-primary) !important;
}

/* Inline code */
.stMarkdown code {
    background-color: var(--bg-subtle) !important;
    color: var(--primary) !important;
    padding: 0.125rem 0.4375rem !important;
    border-radius: var(--radius-xs) !important;
    border: 1px solid var(--border-light) !important;
    font-size: 0.8125rem !important;
}

/* ── 16. DIVIDER ───────────────────────────────────────────── */
hr, .stMarkdown hr {
    border: none !important;
    border-top: 1px solid var(--border-light) !important;
    margin: 1.5rem 0 !important;
}

/* ── 17. CAPTION / SMALL TEXT ──────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-tertiary) !important;
}

/* ── 18. TOAST ─────────────────────────────────────────────── */
[data-testid="stToast"] {
    background-color: var(--white) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* ── 19. SLIDER ────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

.stSlider [data-baseweb="slider"] [data-testid="stTickBar"] {
    background-color: var(--bg-subtle) !important;
}

/* ── 20. CHECKBOX / RADIO ──────────────────────────────────── */
.stCheckbox label span,
.stRadio label span {
    color: var(--text-primary) !important;
}

/* ═════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS — AIRBNB STYLE
   ═════════════════════════════════════════════════════════════════ */

/* ── Hero Section ──────────────────────────────────────────── */
.hero-section {
    text-align: center;
    padding: 3.5rem 0 2rem;
    max-width: 580px;
    margin: 0 auto;
}

.hero-section.animate-in {
    animation: heroFadeIn 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.3125rem 0.875rem;
    background: var(--primary-light);
    color: var(--primary);
    font-size: 0.6875rem;
    font-weight: 600;
    border-radius: var(--radius-full);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-bottom: 1.75rem;
    border: 1px solid var(--primary-border);
}

.hero-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary);
    display: inline-block;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

.hero-title {
    font-family: 'Inter', sans-serif !important;
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.04em !important;
    line-height: 1.06 !important;
    margin-bottom: 1rem !important;
    margin-top: 0 !important;
}

.hero-subtitle {
    font-size: 1.0625rem !important;
    color: var(--text-tertiary) !important;
    line-height: 1.55 !important;
    margin-bottom: 0 !important;
    max-width: 440px;
    margin-left: auto;
    margin-right: auto;
    font-weight: 400;
}

.trust-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1.75rem;
    padding-top: 1.375rem;
    border-top: 1px solid var(--border-light);
}

.trust-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.75rem;
    color: var(--text-tertiary);
    font-weight: 500;
    letter-spacing: 0.005em;
}

.trust-item-icon {
    font-size: 0.875rem;
    opacity: 0.8;
}

/* ── Sidebar Components ────────────────────────────────────── */
.sidebar-brand {
    padding: 0.25rem 0 0.5rem;
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
    width: 36px;
    height: 36px;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, #E61E4D, #D70466);
    color: var(--text-inverse);
    font-size: 1rem;
    flex-shrink: 0;
    box-shadow: var(--shadow-sm);
}

.sidebar-brand-text {
    display: flex;
    flex-direction: column;
    gap: 0;
}

.sidebar-brand-name {
    font-size: 1rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    line-height: 1.1;
}

.sidebar-brand-tag {
    font-size: 0.5625rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 2px;
}

.sidebar-label {
    font-size: 0.5625rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    margin-bottom: 0.375rem !important;
    margin-top: 0.25rem !important;
}

.sidebar-sep {
    height: 1px;
    background: var(--border-light);
    margin: 0.875rem 0;
}

.sidebar-meta {
    padding-top: 0.625rem;
    font-size: 0.5625rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.02em;
}

.sidebar-meta a {
    color: var(--text-tertiary) !important;
    font-weight: 500;
    text-decoration: none !important;
}

.sidebar-meta a:hover {
    color: var(--primary) !important;
}

/* ── Status Indicators ─────────────────────────────────────── */
.indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    font-size: 0.625rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.indicator-active {
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid var(--success-border);
}

.indicator-idle {
    background: var(--bg-subtle);
    color: var(--text-muted);
    border: 1px solid var(--border-light);
}

.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3125rem;
    padding: 0.3125rem 0.75rem;
    background: var(--white);
    color: var(--text-secondary);
    font-size: 0.6875rem;
    font-weight: 500;
    border-radius: var(--radius-full);
    border: 1px solid var(--border-default);
    margin-top: 0.5rem;
    box-shadow: var(--shadow-xs);
}

/* ── Onboarding Card ───────────────────────────────────────── */
.ob-card {
    background: var(--white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-2xl);
    padding: 3rem 2.5rem;
    text-align: center;
    box-shadow: var(--shadow-card);
    max-width: 480px;
    margin: 0 auto;
    animation: cardFadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    transition: box-shadow var(--transition-base);
}

.ob-card:hover {
    box-shadow: var(--shadow-card-hover);
}

.ob-icon {
    width: 56px;
    height: 56px;
    border-radius: var(--radius-lg);
    background: var(--primary-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.375rem;
    margin: 0 auto 1.5rem;
    border: 1px solid var(--primary-border);
}

.ob-heading {
    font-family: 'Inter', sans-serif;
    font-size: 1.3125rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}

.ob-body {
    font-size: 0.9375rem;
    color: var(--text-tertiary);
    line-height: 1.55;
    margin-bottom: 1.75rem;
    max-width: 360px;
    margin-left: auto;
    margin-right: auto;
}

.ob-steps {
    text-align: left;
    background: var(--bg-subtle);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 1.125rem 1.375rem;
    margin-bottom: 0;
}

.ob-steps p {
    margin: 0 !important;
    font-size: 0.8125rem !important;
    color: var(--text-secondary) !important;
    line-height: 2 !important;
    display: flex !important;
    align-items: baseline !important;
    gap: 0.5rem !important;
}

.ob-steps strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.ob-steps a {
    color: var(--primary) !important;
    text-decoration: underline !important;
    text-underline-offset: 2px !important;
}

.ob-steps a:hover {
    opacity: 0.8 !important;
}

.ob-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: var(--radius-full);
    background: var(--primary-light);
    border: 1px solid var(--primary-border);
    color: var(--primary);
    font-size: 0.625rem;
    font-weight: 700;
    flex-shrink: 0;
}

.ob-examples-label {
    font-size: 0.625rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    text-align: center;
    margin-bottom: 0.75rem;
}

.ob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
}

.ob-chip {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.625rem 0.875rem;
    background: var(--white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    font-size: 0.75rem;
    color: var(--text-secondary);
    transition: all var(--transition-base);
    cursor: default;
    font-weight: 500;
}

.ob-chip:hover {
    border-color: var(--primary);
    background: var(--primary-light);
    color: var(--primary);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}

.ob-chip-icon {
    font-size: 0.875rem;
    flex-shrink: 0;
}

/* ── Ready Banner ──────────────────────────────────────────── */
.ready-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.625rem 1rem;
    background: var(--white);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-full);
    margin-bottom: 1rem;
    font-size: 0.75rem;
    color: var(--text-tertiary);
    font-weight: 400;
    box-shadow: var(--shadow-sm);
    animation: cardFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.ready-bar strong {
    color: var(--text-primary);
    font-weight: 600;
}

.ready-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--success);
    flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}

/* ── Footer ────────────────────────────────────────────────── */
.site-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    margin-top: 4rem;
    border-top: 1px solid var(--border-light);
}

.site-footer-name {
    font-size: 0.8125rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.site-footer-desc {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    margin: 0.1875rem 0 0.625rem;
    font-weight: 400;
}

.site-footer-links {
    font-size: 0.6875rem;
    margin-bottom: 0.625rem;
}

.site-footer-links a {
    color: var(--text-tertiary) !important;
    text-decoration: none !important;
    font-weight: 500;
    transition: color var(--transition-fast) !important;
}

.site-footer-links a:hover {
    color: var(--primary) !important;
}

.site-footer-sep {
    color: var(--text-muted);
    margin: 0 0.375rem;
}

.site-footer-legal {
    font-size: 0.5625rem;
    color: var(--text-muted);
    letter-spacing: 0.015em;
}

/* ── ANIMATIONS ────────────────────────────────────────────── */
@keyframes heroFadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── RESPONSIVE ────────────────────────────────────────────── */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .hero-title { font-size: 2rem !important; }
    .hero-subtitle { font-size: 0.9375rem !important; }
    .hero-section { padding: 2.5rem 0 1.5rem; }
    .ob-grid { grid-template-columns: 1fr; }
    .ob-card { padding: 2.25rem 1.75rem; }
    .trust-row { flex-direction: column; gap: 0.5rem; }
}

@media (max-width: 480px) {
    .hero-title { font-size: 1.625rem !important; }
    h1, .stMarkdown h1 { font-size: 1.375rem !important; }
    .ob-card { padding: 1.75rem 1.25rem; border-radius: var(--radius-lg); }
}

/* ── GLOBAL OVERRIDES ──────────────────────────────────────── */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stMarkdown li, .stMarkdown td, .stMarkdown th,
.element-container, .stText, div[data-testid="stText"] {
    color: var(--text-primary) !important;
}

.stTooltipIcon, div[data-testid="tooltipHoverTarget"] {
    color: var(--text-muted) !important;
}

.stSpinner > div {
    border-color: var(--primary) transparent transparent transparent !important;
}

[data-testid="stEmpty"] {
    color: var(--text-muted) !important;
}

[data-testid="stJson"] {
    background-color: var(--bg-subtle) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
}

.stTable table {
    background-color: var(--white) !important;
}

.stTable th {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-light) !important;
    font-weight: 600 !important;
}

.stTable td {
    color: var(--text-primary) !important;
    border-color: var(--border-light) !important;
}

.stNumberInput > div > div > input {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

.stNumberInput button {
    background-color: var(--bg-subtle) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-default) !important;
}

.stDateInput > div > div > input {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-primary) !important;
}

[data-baseweb="calendar"],
[data-baseweb="datepicker"] {
    background-color: var(--white) !important;
    color: var(--text-primary) !important;
}

.stElementContainer, [data-testid="stElementContainer"] {
    color: var(--text-primary) !important;
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
            instant insights with beautiful visualizations.
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

- Understand the user's question
- Perform accurate data analysis using pandas/numpy
- Return clean, meaningful results
- Create a beautiful Plotly visualization when useful

━━━━━━━━━━━━━━━━━━━━━━━
⚠️ STRICT OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━

- Return ONLY executable Python code
- NO explanations, NO markdown, NO comments
- Do NOT include any import statements (all modules are pre-loaded)
- ALWAYS use the variable `df`
- Final output MUST be stored in: `result`
- If visualization is useful, ALSO create: `fig`

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
- Use template='plotly_white' always (light theme app)
- Use these colors: ['#FF385C', '#00A699', '#FC642D', '#484848', '#767676', '#E07912']
- AUTO-SELECT chart type:
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
- For bar charts, use color_discrete_sequence=['#FF385C']
- For line charts, use color_discrete_sequence=['#FF385C']
- For pie charts, use color_discrete_sequence=['#FF385C', '#00A699', '#FC642D', '#484848', '#767676', '#E07912']

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
        <div class="site-footer-desc">AI-powered data analysis for everyone.</div>
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