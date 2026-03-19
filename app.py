import streamlit as st
import pandas as pd
import numpy as np                          # ← moved to top-level
from openai import OpenAI

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="SheetTalk - Chat With Your Data",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
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
        st.session_state.api_key_set = False          # ← reset when key removed
        st.info("Enter your NVIDIA API key")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        # Only re-read when the file actually changes
        if (
            st.session_state.df is None
            or st.session_state.get("file_name") != uploaded_file.name
        ):
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []        # ← reset chat for new file
        st.success(f"Loaded: {uploaded_file.name}")

# ── Main ────────────────────────────────────────────────────────────
st.title("💬 SheetTalk")
st.caption("Chat with your CSV using NVIDIA AI")


# ── Helper: extract executable code from LLM response ──────────────
def extract_code(raw: str) -> str:
    """
    Strip markdown fences, blank lines and comments,
    then return all lines that look like executable pandas code.
    Falls back to the last non-empty line.
    """
    code = raw.replace("```python", "").replace("```", "").strip()
    lines = [
        l.strip()
        for l in code.split("\n")
        if l.strip() and not l.strip().startswith("#")
    ]

    if not lines:
        raise ValueError("LLM returned empty code")

    # Keep every line that references `df` or is an assignment to a var used later
    df_lines = [l for l in lines if "df" in l]

    # If there's exactly one expression → return it;
    # otherwise join with newlines so multi-step code still works.
    if df_lines:
        return "\n".join(df_lines)

    return lines[-1]


# ── Safe builtins whitelist for eval / exec ─────────────────────────
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


def safe_execute(code: str, df: pd.DataFrame):
    """
    Execute pandas code and return the result.
    • Single expressions  → eval()
    • Multi-line / assignments → exec(), then return the last variable.
    """
    global_ns = {"__builtins__": SAFE_BUILTINS, "df": df, "pd": pd, "np": np}

    # Single expression → eval is fine
    if "\n" not in code and "=" not in code.split("#")[0]:
        return eval(code, global_ns)

    # Multi-line or assignment → use exec, capture the last assigned var
    local_ns: dict = {}
    exec(code, global_ns, local_ns)

    # Return `result` if the code assigned one, else the last local
    if "result" in local_ns:
        return local_ns["result"]
    if local_ns:
        return list(local_ns.values())[-1]
    return global_ns.get("df", df)


# ── Chat loop ──────────────────────────────────────────────────────
if st.session_state.api_key_set and st.session_state.df is not None:

    df = st.session_state.df

    # Show history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            if "code" in msg:
                st.code(msg["code"], language="python")
            if "result" in msg:
                st.write(msg["result"])
            if "chart" in msg:
                try:
                    st.bar_chart(msg["chart"])
                except Exception:
                    pass                              # skip chart if it can't render

    # User input
    if prompt := st.chat_input("Ask about your data..."):

        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )

        sample = df.head(5).to_string()
        dtypes = df.dtypes.to_string()

        full_prompt = f"""
You are a strict Python data analyst.

DataFrame: df

Columns and types:
{dtypes}

Sample rows:
{sample}

RULES:
- Only return pandas / numpy code
- Must use the variable `df`
- No explanation, no markdown, no comments
- If the answer needs more than one line, assign the final answer to `result`

Question:
{prompt}
"""

        try:
            client = st.session_state.client

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "Return ONLY executable pandas code."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0,
                max_tokens=300,
            )

            raw = response.choices[0].message.content
            code = extract_code(raw)
            result = safe_execute(code, df)           # ← uses safe wrapper

            msg: dict = {
                "role": "assistant",
                "code": code,
                "result": result,
            }

            # Attach chart when the result is plottable
            if isinstance(result, pd.Series) and result.dtype.kind in "iufb":
                msg["chart"] = result
            elif isinstance(result, pd.DataFrame):
                num = result.select_dtypes(include="number")
                if not num.empty:
                    msg["chart"] = num

            st.session_state.chat_history.append(msg)

        except Exception as e:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            )

        st.rerun()

# ── Fallback messages ──────────────────────────────────────────────
elif not st.session_state.api_key_set:
    st.info("🔑 Enter your NVIDIA API key in the sidebar.")

elif st.session_state.df is None:
    st.info("📂 Upload a CSV file to get started.")