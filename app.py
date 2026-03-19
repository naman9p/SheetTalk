import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px                          # ← NEW: Plotly support
from openai import OpenAI

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="SheetTalk - Chat With Your Data",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for polish ───────────────────────────────────────────
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 1rem; }
    div[data-testid="stCodeBlock"] { border-radius: 8px; }
    .stPlotlyChart { border-radius: 10px; overflow: hidden; }
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
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    api_key = st.text_input("🔑 NVIDIA API Key", type="password")

    model_choice = st.selectbox(
        "🧠 Model",
        [
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "meta/llama3-70b-instruct",
            "nvidia/nemotron-4-340b-instruct",
        ],
    )

    if api_key:
        st.session_state.api_key_set = True
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )
        st.session_state.client = client
        st.success("✅ NVIDIA API Connected")
    else:
        st.session_state.api_key_set = False
        st.info("Enter your NVIDIA API key")

    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        if (
            st.session_state.df is None
            or st.session_state.get("file_name") != uploaded_file.name
        ):
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []
        st.success(f"✅ Loaded: {uploaded_file.name}")

        # Show quick preview
        with st.expander("👀 Preview Data"):
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
            st.caption(
                f"**{st.session_state.df.shape[0]}** rows × "
                f"**{st.session_state.df.shape[1]}** columns"
            )

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Main ────────────────────────────────────────────────────────────
st.title("💬 SheetTalk")
st.caption("Chat with your CSV using NVIDIA AI • Powered by Plotly visualizations")


# ── Helper: extract executable code from LLM response ──────────────
def extract_code(raw: str) -> str:
    """
    Strip markdown fences, blank lines, and stray comments.
    Returns all executable lines joined together.
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response")

    code = raw.strip()

    # Remove markdown code fences
    if "```" in code:
        parts = code.split("```")
        code_blocks = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # inside a fence
                # Strip optional language tag on first line
                lines = part.split("\n")
                if lines and lines[0].strip().lower() in (
                    "python", "py", "python3", ""
                ):
                    lines = lines[1:]
                code_blocks.append("\n".join(lines))
        if code_blocks:
            code = "\n".join(code_blocks).strip()

    # Remove comment-only lines but keep inline comments
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(line)  # preserve original indentation

    if not lines:
        raise ValueError("LLM returned only comments / empty code")

    return "\n".join(lines)


# ── Safe builtins whitelist ─────────────────────────────────────────
SAFE_BUILTINS = {
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "print": print,
    "isinstance": isinstance,
    "type": type,
    "True": True,
    "False": False,
    "None": None,
}


def safe_execute(code: str, df: pd.DataFrame) -> dict:
    """
    Execute pandas/plotly code and return a dict with:
      - 'result': the computed data
      - 'fig':    a Plotly figure (or None)
    """
    global_ns = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),          # protect original df
        "pd": pd,
        "np": np,
        "px": px,                 # ← Plotly express available
    }

    local_ns: dict = {}

    # Try eval first for single expressions with no assignment
    is_single_expr = (
        "\n" not in code.strip()
        and "=" not in code.split("#")[0]
        and "import" not in code
    )

    if is_single_expr:
        try:
            result = eval(code, global_ns)
            return {"result": result, "fig": None}
        except SyntaxError:
            pass  # fall through to exec

    # Multi-line or assignment → exec
    exec(code, global_ns, local_ns)

    # Extract result
    result = local_ns.get("result", None)
    if result is None:
        # Fallback: last assigned variable (skip fig)
        non_fig = {k: v for k, v in local_ns.items() if k != "fig"}
        if non_fig:
            result = list(non_fig.values())[-1]

    # Extract figure
    fig = local_ns.get("fig", None)

    return {"result": result, "fig": fig}


# ── Build the advanced system prompt ───────────────────────────────
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
- NO explanations
- NO markdown
- NO comments
- ALWAYS use the variable `df`
- Final output MUST be stored in: result
- If visualization is useful, ALSO create: fig

━━━━━━━━━━━━━━━━━━━━━━━
📊 VISUALIZATION RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Use plotly.express as px (already imported)
- Charts must look clean, modern, and professional

AUTO-SELECT chart type:
- Comparison → px.bar
- Trends/time → px.line
- Distribution → px.histogram
- Proportions → px.pie

RULES:
- Always sort values before plotting
- Limit to top 10 categories if too many values
- Add proper title
- Use clear axis labels
- Avoid clutter
- Use template='plotly_white'

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

    # ── Show chat history ──
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            if "code" in msg:
                with st.expander("🔍 View Code", expanded=False):
                    st.code(msg["code"], language="python")
            if "result" in msg and msg["result"] is not None:
                st.write(msg["result"])
            if "fig" in msg and msg["fig"] is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)
            if "chart" in msg:
                try:
                    st.bar_chart(msg["chart"])
                except Exception:
                    pass

    # ── User input ──
    if prompt := st.chat_input("Ask about your data..."):

        # Show user message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        try:
            client = st.session_state.client

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(prompt, df)},
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

            # Fallback bar chart if no plotly fig but result is plottable
            if fig is None:
                if isinstance(result, pd.Series) and result.dtype.kind in "iufb":
                    assistant_msg["chart"] = result
                elif isinstance(result, pd.DataFrame):
                    num = result.select_dtypes(include="number")
                    if not num.empty and len(num) <= 50:
                        assistant_msg["chart"] = num

            st.session_state.chat_history.append(assistant_msg)

        except Exception as e:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            )

        st.rerun()

# ── Fallback messages ──────────────────────────────────────────────
elif not st.session_state.api_key_set:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("🔑 Enter your NVIDIA API key in the sidebar to get started.")
        st.markdown("""
        **How to get your API key:**
        1. Visit [NVIDIA AI](https://build.nvidia.com/)
        2. Create an account / sign in
        3. Generate an API key
        4. Paste it in the sidebar
        """)

elif st.session_state.df is None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("📂 Upload a CSV file in the sidebar to start chatting.")
        st.markdown("""
        **Supported questions:**
        - 📊 *"Show me sales by region"*
        - 📈 *"What's the trend over time?"*
        - 🔢 *"What is the average revenue?"*
        - 🏆 *"Top 10 customers by spend"*
        - 🥧 *"Distribution of categories"*
        """)