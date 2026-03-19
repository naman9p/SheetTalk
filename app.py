import streamlit as st
import pandas as pd
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="SheetTalk - Chat With Your Data",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chatbot UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    
    .main-title {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1rem;
        margin-bottom: 30px;
    }
    
    .stTextInput > div > div > input {
        background: rgba(40, 40, 60, 0.8) !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stFileUploader {
        background: rgba(40, 40, 60, 0.6);
        border-radius: 16px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .result-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    
    .stCodeBlock {
        background: #1e1e2e !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .info-message {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.85rem;
        margin-top: 40px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer span {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
    }
    
    .section-header {
        color: #00d2ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d2ff;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'df_uploaded' not in st.session_state:
    st.session_state.df_uploaded = False
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    model_choice = st.selectbox(
        "🧠 Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model to use"
    )
    
    if api_key:
        st.session_state.api_key_set = True
        client = OpenAI(api_key=api_key)
        st.session_state.client = client
        st.markdown(
            '<div class="success-message">✅ API Key Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-message">ℹ️ Enter your OpenAI API key to get started</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### 📊 Your Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_uploaded = True
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Columns:**")
        st.code(", ".join(df.columns.tolist()), language=None)

# Main content
st.markdown('<h1 class="main-title">💬 SheetTalk</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">✨ AI-Powered Chat with Your CSV Data ✨</p>',
    unsafe_allow_html=True
)

# ─── HELPER: extract executable pandas code from LLM response ───
def extract_code(raw_response: str) -> str:
    """
    Clean the LLM response and return a single executable line
    that operates on `df`.
    """
    # Strip markdown fences
    code = raw_response.replace("```python", "").replace("```", "").strip()

    # Remove common preamble words the model sometimes adds
    lines = code.split("\n")

    # Keep only lines that look like actual code (contain 'df')
    df_lines = [
        line.strip() for line in lines
        if "df" in line
        and not line.strip().startswith("#")
        and not line.strip().lower().startswith("answer")
    ]

    if df_lines:
        return df_lines[-1]          # take the last (most complete) expression

    # Fallback: return the last non-empty, non-comment line
    fallback = [
        line.strip() for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    if fallback:
        return fallback[-1]

    raise ValueError(f"Could not extract executable code from response:\n{raw_response}")


# ─── MAIN CHAT INTERFACE ────────────────────────────────────────
if st.session_state.api_key_set and st.session_state.df_uploaded:
    st.markdown("---")
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Data"])

    with tab1:
        # ── FIX 1: display logic now handles ALL message shapes ──
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    # Show plain text / error if present
                    if "content" in message:
                        st.markdown(message["content"])

                    # Show generated code
                    if "code" in message:
                        st.markdown("**Generated Code:**")
                        st.code(message["code"], language="python")

                    # Show result
                    if "result" in message:
                        st.markdown("**Result:**")
                        result = message["result"]
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                        elif isinstance(result, pd.Series):
                            st.dataframe(
                                result.reset_index(),
                                use_container_width=True
                            )
                        else:
                            st.markdown(f"```\n{result}\n```")

                    # Show chart
                    if "chart_data" in message:
                        st.markdown("**Visualization:**")
                        try:
                            st.bar_chart(message["chart_data"])
                        except Exception:
                            pass   # skip chart if data shape is incompatible

        # Chat input
        if prompt := st.chat_input("💭 Ask anything about your data..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )

            df = st.session_state.df

            # ── FIX 2: improved prompt with sample data ──
            sample_str = df.head(3).to_string()
            dtypes_str = df.dtypes.to_string()

            openai_prompt = f"""You are an expert Python data analyst.

A pandas DataFrame named `df` already exists in memory.

Columns and types:
{dtypes_str}

Sample rows:
{sample_str}

RULES — follow them strictly:
1. Return ONLY a single line of valid Python code.
2. The code must be a pandas expression starting with `df`.
3. Do NOT assign to a variable. Do NOT use print().
4. Do NOT include any explanation, markdown, or comments.
5. The expression must evaluate to a result (not None).

User question: {prompt}
"""

            with st.spinner("🤔 Analyzing your data..."):
                try:
                    client = st.session_state.client
                    completion = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a code-only assistant. "
                                    "Reply with exactly one line of pandas code. "
                                    "No explanation."
                                ),
                            },
                            {"role": "user", "content": openai_prompt},
                        ],
                        temperature=0,
                    )

                    raw = completion.choices[0].message.content

                    # ── FIX 3: robust code extraction ──
                    code = extract_code(raw)

                    # ── FIX 4: safe eval with limited namespace ──
                    import numpy as np
                    result = eval(
                        code,
                        {"__builtins__": {}},
                        {"df": df, "pd": pd, "np": np},
                    )

                    # Guard against None results
                    if result is None:
                        raise ValueError(
                            "The generated code returned None. "
                            "Try rephrasing your question."
                        )

                    response_data = {
                        "role": "assistant",
                        "code": code,
                        "result": result,
                    }

                    # Only attach chart data for numeric series / dataframes
                    if isinstance(result, pd.Series):
                        if pd.api.types.is_numeric_dtype(result):
                            response_data["chart_data"] = result
                    elif isinstance(result, pd.DataFrame):
                        numeric_cols = result.select_dtypes(include="number")
                        if not numeric_cols.empty:
                            response_data["chart_data"] = numeric_cols

                    st.session_state.chat_history.append(response_data)

                except Exception as e:
                    # ── FIX 5: error messages now have "content" key ──
                    #     AND the display loop renders "content" for assistants
                    debug_info = ""
                    if 'raw' in dir():
                        debug_info = f"\n\n**Raw LLM response:**\n```\n{raw}\n```"
                    if 'code' in dir():
                        debug_info += (
                            f"\n\n**Extracted code:**\n```python\n{code}\n```"
                        )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            f"❌ **Error:** {str(e)}{debug_info}\n\n"
                            "💡 Try rephrasing your question."
                        ),
                    })

            st.rerun()

    with tab2:
        df = st.session_state.df
        st.markdown(
            '<div class="section-header">📋 Dataset Preview</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric(
                "Numeric Columns",
                len(df.select_dtypes(include='number').columns)
            )
        with col4:
            st.metric(
                "Text Columns",
                len(df.select_dtypes(include='object').columns)
            )

        st.markdown("### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dtype) for dtype in df.dtypes.values],
            'Non-Null': df.notna().sum().values,
            'Null': df.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

elif not st.session_state.api_key_set:
    st.markdown("""
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
             color: #e0e0e0; padding: 20px; border-radius: 20px;
             border: 1px solid rgba(255, 255, 255, 0.1);">
            👋 Welcome to <b>SheetTalk</b>!<br><br>
            1. Enter your OpenAI API key in the sidebar<br>
            2. Upload a CSV file<br>
            3. Start asking questions about your data!<br><br>
            💡 Get your API key at
            <a href="https://platform.openai.com/api-keys" target="_blank">
            platform.openai.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-around;
         text-align: center; color: #a0a0a0;">
        <div><div style="font-size: 2rem;">📊</div><div>Upload CSV</div></div>
        <div><div style="font-size: 2rem;">💬</div><div>Ask Questions</div></div>
        <div><div style="font-size: 2rem;">🤖</div><div>AI Analysis</div></div>
        <div><div style="font-size: 2rem;">📈</div><div>Visualize</div></div>
    </div>
    """, unsafe_allow_html=True)

elif not st.session_state.df_uploaded:
    st.markdown("""
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
             color: #e0e0e0; padding: 20px; border-radius: 20px;
             border: 1px solid rgba(255, 255, 255, 0.1);">
            👋 API Key connected! Now upload a CSV file in the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
year = pd.Timestamp.now().year
st.markdown(f"""
<div class="footer">
    © {year} SheetTalk. All rights reserved.<br>
    Made with ❤️ for data enthusiasts | Powered by <span>Dhokla</span>
</div>
""", unsafe_allow_html=True)