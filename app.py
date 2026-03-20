import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime
import re
import ast

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="SheetTalk — Chat With Your Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "SheetTalk — AI-powered data analysis by Dhokla."},
)

# ── Chart colors ────────────────────────────────────────────────────
CHART_COLORS = [
    "#1DB954", "#1ED760", "#509BF5", "#F573A0",
    "#E8115B", "#F59B23", "#B49BC8", "#4687D6",
    "#2D46B9", "#E91429",
]


def style_figure(fig):
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=14, color="#B3B3B3"),
        title=dict(font=dict(size=18, color="#FFFFFF", family="Inter, sans-serif"), x=0.01, xanchor="left", pad=dict(l=4, t=6, b=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=56, r=24, t=72, b=56),
        legend=dict(font=dict(size=12, color="#B3B3B3"), bgcolor="rgba(0,0,0,0)", borderwidth=0, orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        colorway=CHART_COLORS,
        hoverlabel=dict(bgcolor="#282828", font_size=13, font_family="Inter, sans-serif", font_color="#FFFFFF", bordercolor="#404040"),
        xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor="#333333", tickfont=dict(size=12, color="#A0A0A0"), title_font=dict(size=13, color="#B3B3B3"), zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.06)", griddash="dot", showline=False, tickfont=dict(size=12, color="#A0A0A0"), title_font=dict(size=13, color="#B3B3B3"), zeroline=False),
        bargap=0.28, bargroupgap=0.08,
    )
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            trace.marker.line.width = 0
            trace.marker.cornerradius = 5
            if not trace.marker.color:
                trace.marker.color = "#1DB954"
            trace.marker.opacity = 0.92
        if trace.type == "scatter" and trace.mode and "lines" in trace.mode:
            trace.line.width = 2.5
            trace.line.shape = "spline"
        if trace.type == "pie":
            trace.marker.line.width = 2.5
            trace.marker.line.color = "#181818"
            trace.textfont = dict(size=12, color="#FFFFFF", family="Inter, sans-serif")
            trace.marker.colors = CHART_COLORS
        if trace.type == "histogram":
            trace.marker.line.width = 1
            trace.marker.line.color = "#181818"
            trace.marker.opacity = 0.88
    return fig


def validate_api_key(api_key_val, model_val):
    try:
        test_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key_val)
        response = test_client.chat.completions.create(model=model_val, messages=[{"role": "user", "content": "Hi"}], max_tokens=5, temperature=0)
        if response and response.choices:
            return {"valid": True, "client": test_client, "error": None}
        return {"valid": False, "client": None, "error": "Empty response"}
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return {"valid": False, "client": None, "error": "Invalid API key"}
        if "404" in error_msg:
            return {"valid": False, "client": None, "error": "Model not found"}
        if "429" in error_msg:
            return {"valid": True, "client": test_client, "error": None}
        return {"valid": False, "client": None, "error": error_msg[:120]}


def sanitize_code(code_str, dataframe):
    columns = list(dataframe.columns)
    for col in columns:
        for q in ["'", '"']:
            bad_pattern = rf'df\[{q}{re.escape(col)}{q}\s*([<>=!]+)\s*([^\]]+)\]'
            good_replacement = rf'df[df[{q}{col}{q}] \1 \2]'
            code_str = re.sub(bad_pattern, good_replacement, code_str)
        str_pattern = rf'df\["({re.escape(col)})\s*([=!<>]+)\s*(.+?)"\]'
        match = re.search(str_pattern, code_str)
        if match:
            col_name, op, val = match.groups()
            code_str = re.sub(str_pattern, f'df[df["{col_name}"] {op} {val.strip()}]', code_str)
    cleaned_lines = []
    for line in code_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def validate_syntax(code_str):
    try:
        ast.parse(code_str)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def extract_code(raw):
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response")
    code_str = raw.strip()
    if "```" in code_str:
        parts = code_str.split("```")
        code_blocks = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                lines = part.split("\n")
                if lines and lines[0].strip().lower() in ("python", "py", "python3", ""):
                    lines = lines[1:]
                code_blocks.append("\n".join(lines))
        if code_blocks:
            code_str = "\n".join(code_blocks).strip()
    final_lines = []
    for line in code_str.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        final_lines.append(line)
    code_str = "\n".join(final_lines).strip()
    if not code_str:
        raise ValueError("LLM returned no executable code")
    return code_str


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


def safe_execute(code_str, dataframe):
    global_ns = {
        "__builtins__": SAFE_BUILTINS,
        "df": dataframe.copy(), "pd": pd, "np": np, "px": px, "go": go,
    }
    local_ns = {}
    try:
        r = eval(code_str, global_ns)
        return {"result": r, "fig": None, "summary": None}
    except SyntaxError:
        pass
    exec(code_str, global_ns, local_ns)
    result = local_ns.get("result", None)
    fig = local_ns.get("fig", None)
    summary = local_ns.get("summary", None)
    if result is None:
        for k, v in local_ns.items():
            if k not in ("fig", "summary") and isinstance(v, (pd.DataFrame, pd.Series)):
                result = v
                break
        if result is None:
            non_special = {k: v for k, v in local_ns.items() if k not in ("fig", "summary")}
            if non_special:
                result = list(non_special.values())[-1]
    if fig is None:
        gfig = global_ns.get("fig", None)
        if isinstance(gfig, go.Figure):
            fig = gfig
    return {"result": result, "fig": fig, "summary": summary}


def generate_fallback_summary(result, question):
    if result is None:
        return "Analysis complete."
    if isinstance(result, pd.DataFrame):
        return f"Found **{len(result):,}** records across **{len(result.columns)}** columns."
    if isinstance(result, pd.Series):
        return f"Result contains **{len(result):,}** entries."
    if isinstance(result, float):
        return f"The result is **{result:,.2f}**."
    if isinstance(result, int):
        return f"The result is **{result:,}**."
    return f"Analysis complete for: *{question}*"


def build_system_prompt(dataframe):
    columns = list(dataframe.columns)
    col_list = ", ".join([f'"{c}"' for c in columns])
    return f"""You are a world-class Python data analyst working inside a Streamlit app.

The dataset is already loaded as a pandas DataFrame named `df`.
The available columns are: [{col_list}]

CRITICAL SYNTAX RULES:

CORRECT filter syntax:
   result = df[df["CGPA"] < 6]
   result = df[df["Branch"] == "CSE"]
   result = df[(df["CGPA"] < 6) & (df["Year"] == 3)]

WRONG — NEVER generate these:
   df["CGPA" < 6]
   df['CGPA' < 6]
   df["CGPA < 6"]
   df[CGPA < 6]

CORRECT sorting:
   result = df.sort_values("CGPA", ascending=False).head(10)

CORRECT groupby:
   result = df.groupby("Branch")["CGPA"].mean().reset_index()

CORRECT value counts:
   result = df["Branch"].value_counts().reset_index()
   result.columns = ["Branch", "Count"]

CHART RULES:
- Use plotly.express (already imported as px)
- Use template="plotly_dark"
- ALWAYS pass the dataframe as the FIRST argument:
  fig = px.bar(result, x="Branch", y="CGPA", title="...", color_discrete_sequence=["#1DB954"], template="plotly_dark")
  fig = px.pie(result, names="Branch", values="Count", title="...", color_discrete_sequence=["#1DB954","#1ED760","#509BF5","#F573A0","#E8115B","#F59B23"], template="plotly_dark")
  fig = px.line(result, x="Year", y="CGPA", title="...", color_discrete_sequence=["#1DB954"], template="plotly_dark")
  fig = px.histogram(df, x="CGPA", title="...", color_discrete_sequence=["#1DB954"], template="plotly_dark")

DOMAIN KNOWLEDGE:
- "at risk" or "risk students" means CGPA < 6.0
- "good students" or "toppers" means CGPA >= 8.0
- "average" means CGPA between 6.0 and 8.0
- "branch wise" or "department wise" means groupby the branch/department column
- "year wise" means groupby year column
- "top N" means sort descending and take head(N)
- "bottom N" means sort ascending and take head(N)

OUTPUT FORMAT:
- Store the final answer in: result
- Store the chart (if any) in: fig
- Store a 1-2 sentence summary in: summary

STRICT RULES:
- Return ONLY executable Python code
- NO markdown, NO explanations, NO comments
- NO import statements
- Use ONLY the exact column names from the dataset
- Always assign to result, optionally fig and summary
- Handle missing values with .dropna() if needed
- Round numbers: .round(2)
- Sort data before plotting. Limit categories to top 10-15 if too many."""


def build_user_prompt(question, dataframe):
    dtypes = dataframe.dtypes.to_string()
    sample = dataframe.head(5).to_string()
    shape = f"{dataframe.shape[0]} rows x {dataframe.shape[1]} columns"
    columns = ", ".join(dataframe.columns.tolist())
    return f"""Dataset shape: {shape}
Columns: {columns}

Column types:
{dtypes}

Sample (first 5 rows):
{sample}

Question: {question}

Generate Python code to answer this. Use df as the dataframe. Assign answer to result, chart to fig, summary to summary."""


# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
    --black: #000000;
    --bg-darkest: #0f0f0f;
    --bg-dark: #121212;
    --bg-card: #181818;
    --bg-elevated: #1e1e1e;
    --bg-hover: #252525;
    --bg-input: #2a2a2a;
    --bg-highlight: #333333;
    --border-subtle: #282828;
    --border-default: #333333;
    --border-hover: #444444;
    --border-focus: #1DB954;
    --green: #1DB954;
    --green-hover: #1ED760;
    --green-dim: rgba(29, 185, 84, 0.12);
    --green-glow: rgba(29, 185, 84, 0.25);
    --green-border: rgba(29, 185, 84, 0.3);
    --red: #E91429;
    --red-dim: rgba(233, 20, 41, 0.12);
    --red-border: rgba(233, 20, 41, 0.25);
    --text-primary: #FFFFFF;
    --text-secondary: #B3B3B3;
    --text-tertiary: #A0A0A0;
    --text-muted: #6A6A6A;
    --text-dim: #535353;
    --text-inverse: #000000;
    --success: #1DB954;
    --success-bg: rgba(29, 185, 84, 0.1);
    --success-border: rgba(29, 185, 84, 0.2);
    --error: #E91429;
    --error-bg: rgba(233, 20, 41, 0.1);
    --radius-xs: 4px;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --radius-2xl: 24px;
    --radius-pill: 9999px;
    --shadow-xs: 0 1px 3px rgba(0,0,0,0.4);
    --shadow-sm: 0 2px 6px rgba(0,0,0,0.5);
    --shadow-md: 0 4px 14px rgba(0,0,0,0.5);
    --shadow-lg: 0 8px 28px rgba(0,0,0,0.6);
    --shadow-xl: 0 16px 48px rgba(0,0,0,0.65);
    --shadow-card: 0 4px 16px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 20px rgba(29,185,84,0.15);
    --transition-fast: 120ms ease;
    --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}
.stApp {
    background: linear-gradient(180deg, var(--bg-darkest) 0%, var(--bg-dark) 100%) !important;
    min-height: 100vh;
}
.main .block-container {
    max-width: 900px !important;
    padding: 1.5rem 2rem 6rem !important;
}
header[data-testid="stHeader"] {
    background: rgba(15, 15, 15, 0.88) !important;
    backdrop-filter: saturate(180%) blur(24px) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg-highlight); border-radius: var(--radius-pill); }
::selection { background: var(--green-dim); color: var(--text-primary); }

h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin-top: 0 !important;
}
h1, .stMarkdown h1 { font-size: 2rem !important; font-weight: 800 !important; margin-bottom: 0.5rem !important; }
h2, .stMarkdown h2 { font-size: 1.5rem !important; }
h3, .stMarkdown h3 { font-size: 1.1875rem !important; font-weight: 600 !important; }
p, .stMarkdown p, .stText {
    font-size: 1rem !important;
    line-height: 1.7 !important;
    color: var(--text-secondary) !important;
}
a { color: var(--green) !important; text-decoration: none !important; transition: all var(--transition-fast) !important; }
a:hover { color: var(--green-hover) !important; text-decoration: underline !important; }

[data-testid="stSidebar"] { background-color: var(--black) !important; border-right: 1px solid var(--border-subtle) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.25rem 1rem !important; background-color: var(--black) !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { background-color: var(--black) !important; }
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.9375rem !important; color: var(--text-secondary) !important; }
[data-testid="stSidebar"] hr { border: none !important; border-top: 1px solid var(--border-subtle) !important; margin: 0.875rem 0 !important; }
[data-testid="stSidebar"] .stExpander { border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important; background: var(--bg-card) !important; }
[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] { background: var(--bg-card) !important; }

.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background-color: var(--green) !important; color: var(--text-inverse) !important;
    border: none !important; border-radius: var(--radius-pill) !important;
    padding: 0.75rem 2rem !important; font-size: 0.9375rem !important; font-weight: 700 !important;
    transition: all var(--transition-base) !important;
}
.stButton > button[kind="primary"]:hover { background-color: var(--green-hover) !important; transform: scale(1.03) !important; box-shadow: var(--shadow-glow) !important; }
.stButton > button[kind="primary"]:active { transform: scale(0.97) !important; }

.stButton > button[kind="secondary"], .stButton > button[data-testid="baseButton-secondary"], .stButton > button {
    background-color: transparent !important; color: var(--text-primary) !important;
    border: 1px solid var(--border-hover) !important; border-radius: var(--radius-pill) !important;
    padding: 0.625rem 1.5rem !important; font-size: 0.9375rem !important; font-weight: 600 !important;
    transition: all var(--transition-base) !important;
}
.stButton > button[kind="secondary"]:hover, .stButton > button:hover {
    background-color: var(--bg-hover) !important; border-color: var(--text-primary) !important; transform: scale(1.02) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

.stDownloadButton > button {
    background-color: transparent !important; color: var(--text-primary) !important;
    border: 1px solid var(--border-default) !important; border-radius: var(--radius-pill) !important;
    font-size: 0.9375rem !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover { border-color: var(--text-primary) !important; background-color: var(--bg-hover) !important; }

.stTextInput > div > div > input {
    background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important; padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important; color: var(--text-primary) !important;
    transition: all var(--transition-base) !important; caret-color: var(--green) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px var(--green-dim), 0 0 12px var(--green-dim) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

.stTextArea > div > div > textarea {
    background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important; padding: 0.75rem 1rem !important;
    font-size: 0.9375rem !important; color: var(--text-primary) !important;
    caret-color: var(--green) !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: var(--green) !important; box-shadow: 0 0 0 3px var(--green-dim) !important;
}

.stSelectbox > div > div {
    background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-sm) !important; transition: all var(--transition-base) !important;
}
.stSelectbox > div > div:hover { border-color: var(--border-hover) !important; }
.stSelectbox > div > div[aria-expanded="true"] { border-color: var(--green) !important; box-shadow: 0 0 0 3px var(--green-dim) !important; }
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div,
.stSelectbox [data-baseweb="select"] > div > div > div {
    color: var(--text-primary) !important; font-family: 'Inter', sans-serif !important; font-size: 0.9375rem !important;
}
.stSelectbox [data-baseweb="select"] svg { fill: var(--text-secondary) !important; }

[data-baseweb="popover"] { border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important; box-shadow: var(--shadow-xl) !important; }
[data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"] { background-color: var(--bg-elevated) !important; }
[data-baseweb="menu"] [role="option"], [role="listbox"] [role="option"],
[data-baseweb="menu"] li, [role="listbox"] li {
    background-color: var(--bg-elevated) !important; color: var(--text-primary) !important;
    font-size: 0.9375rem !important; padding: 0.5625rem 0.875rem !important;
}
[data-baseweb="menu"] [role="option"]:hover, [role="listbox"] [role="option"]:hover,
[data-baseweb="menu"] li:hover, [role="listbox"] li:hover,
[data-baseweb="menu"] [data-highlighted], [data-baseweb="menu"] [aria-selected="true"],
[role="listbox"] [aria-selected="true"] { background-color: var(--bg-hover) !important; }

.stMultiSelect [data-baseweb="select"] { background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important; border-radius: var(--radius-sm) !important; }
.stMultiSelect [data-baseweb="tag"] { background-color: var(--green-dim) !important; color: var(--green) !important; border: 1px solid var(--green-border) !important; }

.stTextInput > label, .stTextArea > label, .stSelectbox > label,
.stMultiSelect > label, .stFileUploader > label, .stSlider > label,
.stCheckbox > label, .stRadio > label {
    font-size: 0.875rem !important; font-weight: 600 !important; color: var(--text-secondary) !important; margin-bottom: 0.375rem !important;
}

[data-testid="stFileUploader"] {
    background-color: var(--bg-card) !important; border: 2px dashed var(--border-default) !important;
    border-radius: var(--radius-md) !important; padding: 1.25rem !important;
    transition: all var(--transition-base) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green) !important; background-color: var(--green-dim) !important; }
[data-testid="stFileUploader"] section { background-color: transparent !important; }
[data-testid="stFileUploader"] section > button {
    background-color: var(--green) !important; color: var(--text-inverse) !important;
    border: none !important; border-radius: var(--radius-pill) !important;
    font-weight: 700 !important; font-size: 0.8125rem !important;
}
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: var(--text-muted) !important; font-size: 0.8125rem !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] { color: var(--text-primary) !important; }

[data-testid="stMetric"] {
    background-color: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important; padding: 1.125rem 1.375rem !important;
    transition: all var(--transition-base) !important;
}
[data-testid="stMetric"]:hover { box-shadow: var(--shadow-md) !important; transform: translateY(-2px) !important; }
[data-testid="stMetric"] label { font-size: 0.8125rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; color: var(--text-muted) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: var(--green) !important; }

[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important;
    overflow: hidden !important; box-shadow: var(--shadow-sm) !important;
}
[data-testid="stDataFrame"] [data-testid="glideDataEditor"], .dvn-scroller { background-color: var(--bg-card) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important;
    background: var(--bg-card) !important; margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] details { border: none !important; background: var(--bg-card) !important; }
[data-testid="stExpander"] summary {
    padding: 0.75rem 1rem !important; font-weight: 600 !important; font-size: 0.9375rem !important;
    color: var(--text-primary) !important; background: var(--bg-card) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--bg-hover) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 1rem 1rem !important; border-top: 1px solid var(--border-subtle) !important; background: var(--bg-card) !important; }
[data-testid="stExpander"] summary svg { color: var(--text-secondary) !important; }

[data-testid="stAlert"] { border-radius: var(--radius-md) !important; font-size: 0.9375rem !important; background-color: var(--bg-card) !important; color: var(--text-primary) !important; }

.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border-subtle) !important; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.9375rem !important; font-weight: 500 !important; color: var(--text-muted) !important;
    padding: 0.625rem 1.125rem !important; border: none !important; background-color: transparent !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; font-weight: 700 !important; border-bottom-color: var(--green) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--green) !important; }

.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--green), var(--green-hover)) !important; border-radius: var(--radius-pill) !important; }
.stProgress > div > div { background-color: var(--bg-highlight) !important; border-radius: var(--radius-pill) !important; height: 4px !important; }

[data-testid="stChatMessage"] {
    border-radius: var(--radius-lg) !important; padding: 1.375rem 1.5rem !important;
    margin-bottom: 1rem !important; transition: all var(--transition-fast) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: var(--bg-elevated) !important; border: 1px solid var(--border-subtle) !important; }
[data-testid="stChatMessage"]:hover { border-color: var(--border-default) !important; box-shadow: var(--shadow-sm) !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div { color: var(--text-primary) !important; }
[data-testid="stChatMessage"] .stMarkdown p { color: var(--text-secondary) !important; font-size: 1rem !important; line-height: 1.7 !important; }

[data-testid="stChatInput"] { border-top: 1px solid var(--border-subtle) !important; background: var(--bg-darkest) !important; padding-top: 1rem !important; }
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important; font-size: 1.0625rem !important;
    border-radius: var(--radius-pill) !important; border: 2px solid var(--border-default) !important;
    padding: 0.9375rem 1.5rem !important; min-height: 54px !important;
    background: var(--bg-elevated) !important; color: var(--text-primary) !important;
    box-shadow: var(--shadow-sm) !important; transition: all var(--transition-smooth) !important;
    caret-color: var(--green) !important; line-height: 1.4 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; font-size: 1.0625rem !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 4px var(--green-dim), 0 0 24px var(--green-dim), var(--shadow-md) !important;
}
[data-testid="stChatInput"] button {
    background-color: var(--green) !important; border-radius: var(--radius-pill) !important;
    color: var(--text-inverse) !important; border: none !important;
    min-width: 44px !important; min-height: 44px !important;
    transition: all var(--transition-base) !important;
}
[data-testid="stChatInput"] button:hover { background-color: var(--green-hover) !important; transform: scale(1.08) !important; box-shadow: var(--shadow-glow) !important; }
[data-testid="stChatInput"] button:active { transform: scale(0.95) !important; }

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] { background-color: var(--bg-highlight) !important; }
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] { background: var(--green) !important; }

[data-testid="stPlotlyChart"], .stPlotlyChart {
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important; padding: 0.75rem !important;
    box-shadow: var(--shadow-card) !important; transition: all var(--transition-base) !important;
}
[data-testid="stPlotlyChart"]:hover, .stPlotlyChart:hover { box-shadow: var(--shadow-lg) !important; }

div[data-testid="stCodeBlock"] { border-radius: var(--radius-md) !important; border: 1px solid var(--border-default) !important; }
div[data-testid="stCodeBlock"] pre { background-color: var(--bg-darkest) !important; }
code { font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace !important; font-size: 0.875rem !important; }
.stMarkdown code { background-color: var(--bg-elevated) !important; color: var(--green) !important; padding: 0.15rem 0.45rem !important; border-radius: var(--radius-xs) !important; border: 1px solid var(--border-subtle) !important; }

hr, .stMarkdown hr { border: none !important; border-top: 1px solid var(--border-subtle) !important; margin: 1.5rem 0 !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; font-size: 0.875rem !important; }

.stTable table { background-color: var(--bg-card) !important; }
.stTable th { background-color: var(--bg-elevated) !important; color: var(--text-primary) !important; border-color: var(--border-subtle) !important; font-weight: 600 !important; }
.stTable td { color: var(--text-primary) !important; border-color: var(--border-subtle) !important; }

.stNumberInput > div > div > input { background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important; color: var(--text-primary) !important; border-radius: var(--radius-sm) !important; }
.stNumberInput button { background-color: var(--bg-elevated) !important; color: var(--text-primary) !important; border-color: var(--border-default) !important; }
.stDateInput > div > div > input { background-color: var(--bg-input) !important; border: 1px solid var(--border-default) !important; color: var(--text-primary) !important; }

.stSlider [data-baseweb="slider"] [role="slider"] { background-color: var(--green) !important; border-color: var(--green) !important; }
.stCheckbox label span, .stRadio label span { color: var(--text-primary) !important; font-size: 0.9375rem !important; }

.hero-section { text-align: center; padding: 3.5rem 0 2rem; max-width: 620px; margin: 0 auto; }
.hero-section.animate-in { animation: heroFadeIn 0.7s cubic-bezier(0.16, 1, 0.3, 1); }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.375rem 1rem; background: var(--green-dim); color: var(--green);
    font-size: 0.75rem; font-weight: 700; border-radius: var(--radius-pill);
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 1.75rem;
    border: 1px solid var(--green-border);
}
.hero-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }
.hero-title {
    font-family: 'Inter', sans-serif !important; font-size: 3.125rem !important;
    font-weight: 900 !important; color: var(--text-primary) !important;
    letter-spacing: -0.045em !important; line-height: 1.05 !important;
    margin-bottom: 1.125rem !important;
}
.hero-subtitle {
    font-size: 1.1875rem !important; color: var(--text-tertiary) !important;
    line-height: 1.55 !important; max-width: 480px; margin: 0 auto; font-weight: 400;
}
.trust-row {
    display: flex; align-items: center; justify-content: center; gap: 1.75rem;
    margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border-subtle);
}
.trust-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.875rem; color: var(--text-muted); font-weight: 500; }
.trust-item-icon { font-size: 1rem; opacity: 0.7; }

.sidebar-brand { padding: 0.375rem 0 0.5rem; }
.sidebar-brand-inner { display: flex; align-items: center; gap: 0.75rem; }
.sidebar-brand-icon {
    display: flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: var(--radius-sm);
    background: var(--green); color: var(--text-inverse); font-size: 1.125rem;
    box-shadow: var(--shadow-glow);
}
.sidebar-brand-name { font-size: 1.125rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; line-height: 1.1; }
.sidebar-brand-tag { font-size: 0.6875rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-top: 2px; }
.sidebar-label {
    font-size: 0.6875rem !important; font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--text-dim) !important;
    margin-bottom: 0.5rem !important; margin-top: 0.25rem !important;
    display: flex !important; align-items: center !important; gap: 0.375rem !important;
}
.sidebar-label-icon { font-size: 0.8125rem; opacity: 0.6; }
.sidebar-sep { height: 1px; background: var(--border-subtle); margin: 0.875rem 0; }
.sidebar-meta { padding-top: 0.75rem; font-size: 0.75rem; color: var(--text-dim); text-align: center; }
.sidebar-meta a { color: var(--text-muted) !important; font-weight: 500; }
.sidebar-meta a:hover { color: var(--green) !important; }

.indicator {
    display: inline-flex; align-items: center; gap: 0.375rem; padding: 0.3125rem 0.75rem;
    border-radius: var(--radius-pill); font-size: 0.75rem; font-weight: 600;
    margin-top: 0.25rem; transition: all var(--transition-base);
}
.indicator-active { background: var(--success-bg); color: var(--green); border: 1px solid var(--success-border); }
.indicator-error { background: var(--error-bg); color: var(--error); border: 1px solid var(--red-border); }
.indicator-idle { background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border-subtle); }
.indicator-checking { background: var(--bg-elevated); color: var(--text-tertiary); border: 1px solid var(--border-default); animation: checking-pulse 1.5s ease-in-out infinite; }
@keyframes checking-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.validation-msg { font-size: 0.75rem; margin-top: 0.375rem; padding: 0.3125rem 0.625rem; border-radius: var(--radius-xs); font-weight: 500; }
.validation-error { color: var(--error); background: var(--red-dim); }

.file-pill {
    display: inline-flex; align-items: center; gap: 0.3125rem; padding: 0.375rem 0.875rem;
    background: var(--bg-card); color: var(--text-secondary); font-size: 0.8125rem; font-weight: 500;
    border-radius: var(--radius-pill); border: 1px solid var(--border-default); margin-top: 0.5rem;
}

.summary-card {
    background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--green); border-radius: var(--radius-md);
    padding: 0.875rem 1.125rem; margin-bottom: 0.75rem;
    font-size: 0.9375rem; color: var(--text-secondary); line-height: 1.6;
    animation: cardFadeIn 0.3s ease;
}
.summary-card strong { color: var(--text-primary); }

.ob-card {
    background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-2xl);
    padding: 3.25rem 2.75rem; text-align: center; box-shadow: var(--shadow-card);
    max-width: 520px; margin: 0 auto; animation: cardFadeIn 0.5s ease;
    transition: all var(--transition-base);
}
.ob-card:hover { box-shadow: var(--shadow-lg); border-color: var(--border-default); }
.ob-icon {
    width: 64px; height: 64px; border-radius: var(--radius-lg); background: var(--green-dim);
    display: flex; align-items: center; justify-content: center; font-size: 1.625rem;
    margin: 0 auto 1.75rem; border: 1px solid var(--green-border);
}
.ob-heading { font-size: 1.5rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.03em; margin-bottom: 0.625rem; }
.ob-body { font-size: 1.0625rem; color: var(--text-tertiary); line-height: 1.55; margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto; }
.ob-steps {
    text-align: left; background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md); padding: 1.25rem 1.5rem;
}
.ob-steps p { margin: 0 !important; font-size: 1rem !important; color: var(--text-secondary) !important; line-height: 2.2 !important; display: flex !important; align-items: baseline !important; gap: 0.625rem !important; }
.ob-steps strong { color: var(--text-primary) !important; font-weight: 600 !important; }
.ob-steps a { color: var(--green) !important; }
.ob-step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 24px; height: 24px; border-radius: var(--radius-pill);
    background: var(--green-dim); border: 1px solid var(--green-border);
    color: var(--green); font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
}
.ob-examples-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-dim); text-align: center; margin-bottom: 0.875rem; }
.ob-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.ob-chip {
    display: flex; align-items: center; gap: 0.4rem; padding: 0.75rem 1rem;
    background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md); font-size: 0.875rem; color: var(--text-secondary);
    transition: all var(--transition-base); cursor: default; font-weight: 500;
}
.ob-chip:hover { border-color: var(--green); background: var(--green-dim); color: var(--green); transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.ob-chip-icon { font-size: 1rem; }

.ready-bar {
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    padding: 0.75rem 1.25rem; background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill); margin-bottom: 1.25rem; font-size: 0.875rem;
    color: var(--text-tertiary); box-shadow: var(--shadow-sm); animation: cardFadeIn 0.35s ease;
}
.ready-bar strong { color: var(--text-primary); font-weight: 600; }
.ready-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse-dot 2s ease-in-out infinite; box-shadow: 0 0 6px var(--green-dim); }

.site-footer { text-align: center; padding: 3rem 0 1.5rem; margin-top: 4.5rem; border-top: 1px solid var(--border-subtle); }
.site-footer-name { font-size: 1.0625rem; font-weight: 800; color: var(--text-primary); letter-spacing: -0.02em; }
.site-footer-desc { font-size: 0.9375rem; color: var(--text-tertiary); margin: 0.3125rem 0 0.875rem; font-weight: 400; }
.site-footer-links { font-size: 0.875rem; margin-bottom: 0.875rem; }
.site-footer-links a { color: var(--text-muted) !important; font-weight: 500; }
.site-footer-links a:hover { color: var(--green) !important; }
.site-footer-sep { color: var(--text-dim); margin: 0 0.5rem; }
.site-footer-legal { font-size: 0.8125rem; color: var(--text-muted); letter-spacing: 0.01em; }

@keyframes heroFadeIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes cardFadeIn { from { opacity: 0; transform: translateY(12px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }

@media (max-width: 768px) {
    .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .hero-title { font-size: 2.25rem !important; }
    .hero-subtitle { font-size: 1rem !important; }
    .hero-section { padding: 2.5rem 0 1.5rem; }
    .ob-grid { grid-template-columns: 1fr; }
    .ob-card { padding: 2.25rem 1.75rem; }
    .trust-row { flex-direction: column; gap: 0.5rem; }
}
@media (max-width: 480px) {
    .hero-title { font-size: 1.875rem !important; }
    h1, .stMarkdown h1 { font-size: 1.5rem !important; }
    .ob-card { padding: 1.75rem 1.25rem; border-radius: var(--radius-lg); }
}

.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown td, .stMarkdown th,
.element-container, .stText, div[data-testid="stText"] { color: var(--text-primary) !important; }
.stSpinner > div { border-color: var(--green) transparent transparent transparent !important; }
[data-testid="stEmpty"] { color: var(--text-dim) !important; }
[data-testid="stJson"] { background-color: var(--bg-card) !important; border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important; }
[data-testid="stToast"] { background-color: var(--bg-card) !important; color: var(--text-primary) !important; border: 1px solid var(--border-default) !important; border-radius: var(--radius-md) !important; box-shadow: var(--shadow-lg) !important; }
.stElementContainer, [data-testid="stElementContainer"] { color: var(--text-primary) !important; }
.css-1d391kg, .css-12w0qpk, .css-1avcm0n, .css-18e3th9, .css-hxt7ib { background-color: var(--bg-dark) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────
for key, default in [
    ("chat_history", []), ("df", None), ("api_key_set", False),
    ("api_validated", False), ("api_error", None), ("last_api_key", ""),
    ("file_name", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><div class="sidebar-brand-inner">'
        '<div class="sidebar-brand-icon">📊</div>'
        '<div style="display:flex;flex-direction:column;gap:1px">'
        '<span class="sidebar-brand-name">SheetTalk</span>'
        '<span class="sidebar-brand-tag">AI Data Analyst</span>'
        '</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label"><span class="sidebar-label-icon">🔌</span> Connection</p>', unsafe_allow_html=True)

    api_key = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-…", label_visibility="collapsed", help="Get your key at build.nvidia.com")
    model_choice = st.selectbox("Model", [
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "meta/llama3-70b-instruct",
        "nvidia/nemotron-4-340b-instruct",
    ], label_visibility="collapsed")

    if api_key:
        key_changed = api_key != st.session_state.last_api_key
        if key_changed:
            st.session_state.last_api_key = api_key
            st.session_state.api_validated = False
            st.session_state.api_error = None
            st.session_state.api_key_set = False
            st.markdown('<div class="indicator indicator-checking">⏳ Validating…</div>', unsafe_allow_html=True)
            with st.spinner("Testing API key…"):
                validation_result = validate_api_key(api_key, model_choice)
            if validation_result["valid"]:
                st.session_state.api_key_set = True
                st.session_state.api_validated = True
                st.session_state.api_error = None
                st.session_state.client = validation_result["client"]
                st.rerun()
            else:
                st.session_state.api_key_set = False
                st.session_state.api_validated = False
                st.session_state.api_error = validation_result["error"]
                st.rerun()
        else:
            if st.session_state.api_validated:
                st.markdown('<div class="indicator indicator-active">✓ Connected</div>', unsafe_allow_html=True)
            elif st.session_state.api_error:
                st.markdown('<div class="indicator indicator-error">✗ Connection failed</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="validation-msg validation-error">{st.session_state.api_error}</div>', unsafe_allow_html=True)
    else:
        st.session_state.api_key_set = False
        st.session_state.api_validated = False
        st.session_state.api_error = None
        st.session_state.last_api_key = ""
        st.markdown('<div class="indicator indicator-idle">○ Awaiting key</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label"><span class="sidebar-label-icon">📁</span> Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], key="file_upload", label_visibility="collapsed")

    if uploaded_file is not None:
        if st.session_state.df is None or st.session_state.get("file_name") != uploaded_file.name:
            try:
                if uploaded_file.name.endswith((".xlsx", ".xls")):
                    loaded_df = pd.read_excel(uploaded_file)
                else:
                    loaded_df = pd.read_csv(uploaded_file)
                st.session_state.df = loaded_df
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []
            except Exception as load_err:
                st.error(f"Could not read file: {load_err}")

        if st.session_state.df is not None:
            st.markdown(f'<div class="file-pill">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)
            with st.expander("Preview data", expanded=False):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                st.caption(f"**{st.session_state.df.shape[0]:,}** rows × **{st.session_state.df.shape[1]}** columns")

    st.markdown('<div class="sidebar-sep"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label"><span class="sidebar-label-icon">⚙️</span> Controls</p>', unsafe_allow_html=True)

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<div class="sidebar-meta">v2.0 · <a href="#">Docs</a> · <a href="#">GitHub</a></div>', unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-section animate-in">'
    '<div class="hero-badge"><span class="hero-badge-dot"></span> AI-Powered Analysis</div>'
    '<h1 class="hero-title">Chat with your data.</h1>'
    '<p class="hero-subtitle">Upload a CSV or Excel file, ask questions in plain English, and get instant insights with beautiful visualizations.</p>'
    '<div class="trust-row">'
    '<span class="trust-item"><span class="trust-item-icon">🔒</span> Session-only processing</span>'
    '<span class="trust-item"><span class="trust-item-icon">⚡</span> Real-time analysis</span>'
    '<span class="trust-item"><span class="trust-item-icon">📊</span> Auto-visualizations</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ── Chat loop ──────────────────────────────────────────────────────
if st.session_state.api_key_set and st.session_state.df is not None:
    df = st.session_state.df

    st.markdown(
        f'<div class="ready-bar">'
        f'<span class="ready-dot"></span>'
        f'Ready — analyzing <strong>{st.session_state.get("file_name", "your file")}</strong>'
        f'&nbsp;·&nbsp; {df.shape[0]:,} rows × {df.shape[1]} columns'
        f'</div>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            if "summary" in msg and msg["summary"]:
                st.markdown(f'<div class="summary-card">{msg["summary"]}</div>', unsafe_allow_html=True)
            if "code" in msg:
                with st.expander("View generated code", expanded=False):
                    st.code(msg["code"], language="python")
            if "result" in msg and msg["result"] is not None:
                result_val = msg["result"]
                if isinstance(result_val, pd.DataFrame):
                    st.dataframe(result_val, use_container_width=True)
                elif isinstance(result_val, pd.Series):
                    st.dataframe(result_val.to_frame(), use_container_width=True)
                else:
                    st.write(result_val)
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(style_figure(msg["fig"]), use_container_width=True)
            if "chart" in msg and msg["chart"] is not None:
                try:
                    st.bar_chart(msg["chart"])
                except Exception:
                    pass

    if prompt := st.chat_input("Ask anything about your data…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        try:
            client = st.session_state.client

            with st.spinner("Analyzing your data…"):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": build_system_prompt(df)},
                        {"role": "user", "content": build_user_prompt(prompt, df)},
                    ],
                    temperature=0,
                    max_tokens=1500,
                )

            raw = response.choices[0].message.content
            code = extract_code(raw)
            code = sanitize_code(code, df)

            is_valid, syntax_err = validate_syntax(code)
            if not is_valid:
                raise SyntaxError(f"Generated code has syntax errors: {syntax_err}")

            output = safe_execute(code, df)
            result = output["result"]
            fig = output["fig"]
            summary = output["summary"]

            if not summary:
                summary = generate_fallback_summary(result, prompt)

            assistant_msg = {"role": "assistant", "code": code, "summary": summary}

            if result is not None:
                assistant_msg["result"] = result
            if fig is not None:
                assistant_msg["fig"] = fig

            if fig is None and result is not None:
                if isinstance(result, pd.Series) and result.dtype.kind in "iufb":
                    assistant_msg["chart"] = result
                elif isinstance(result, pd.DataFrame):
                    num = result.select_dtypes(include="number")
                    if not num.empty and len(num) <= 50:
                        assistant_msg["chart"] = num

            st.session_state.chat_history.append(assistant_msg)

        except Exception as e:
            error_text = str(e)
            if "SyntaxError" in error_text or "syntax" in error_text.lower():
                friendly = "⚠️ The AI generated code with a syntax error. Please try rephrasing your question."
            elif "KeyError" in error_text:
                friendly = f"⚠️ Column not found in your dataset. Please check column names.\n\n`{error_text}`"
            elif "NameError" in error_text:
                friendly = f"⚠️ A variable reference error occurred. Try asking differently.\n\n`{error_text}`"
            else:
                friendly = f"⚠️ Something went wrong. Try rephrasing your question.\n\n`{error_text[:200]}`"

            st.session_state.chat_history.append({"role": "assistant", "content": friendly})

        st.rerun()

elif not st.session_state.api_key_set:
    st.markdown(
        '<div class="ob-card">'
        '<div class="ob-icon">🔑</div>'
        '<div class="ob-heading">Connect your NVIDIA API</div>'
        '<div class="ob-body">Enter your API key in the sidebar to unlock AI-powered data analysis and visualization.</div>'
        '<div class="ob-steps">'
        '<p><span class="ob-step-num">1</span> Visit <a href="https://build.nvidia.com/" target="_blank">build.nvidia.com</a></p>'
        '<p><span class="ob-step-num">2</span> Create an account or sign in</p>'
        '<p><span class="ob-step-num">3</span> Generate an API key</p>'
        '<p><span class="ob-step-num">4</span> Paste it in the sidebar</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )

elif st.session_state.df is None:
    st.markdown(
        '<div class="ob-card">'
        '<div class="ob-icon">📂</div>'
        '<div class="ob-heading">Upload your dataset</div>'
        '<div class="ob-body">Drop a CSV or Excel file in the sidebar to start exploring your data with natural language.</div>'
        '<div class="ob-examples-label">Example questions you can ask</div>'
        '<div class="ob-grid">'
        '<div class="ob-chip"><span class="ob-chip-icon">📊</span> Show at-risk students</div>'
        '<div class="ob-chip"><span class="ob-chip-icon">📈</span> Average CGPA by branch</div>'
        '<div class="ob-chip"><span class="ob-chip-icon">🔢</span> Top 10 by revenue</div>'
        '<div class="ob-chip"><span class="ob-chip-icon">🏆</span> Count by category</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

# ── Footer ──────────────────────────────────────────────────────────
year = datetime.now().year
st.markdown(
    f'<div class="site-footer">'
    f'<div class="site-footer-name">SheetTalk</div>'
    f'<div class="site-footer-desc">AI-powered data analysis for everyone.</div>'
    f'<div class="site-footer-links">'
    f'<a href="#">Privacy</a><span class="site-footer-sep">·</span>'
    f'<a href="#">Terms</a><span class="site-footer-sep">·</span>'
    f'<a href="#">GitHub</a>'
    f'</div>'
    f'<div class="site-footer-legal">© {year} SheetTalk · Powered by Dhokla</div>'
    f'</div>',
    unsafe_allow_html=True,
)