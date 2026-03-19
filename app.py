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
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    
    /* Title Styling */
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
    
    /* API Key Input */
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
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(40, 40, 60, 0.6);
        border-radius: 16px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
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
    
    /* Result Container */
    .result-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    
    /* Code Display */
    .stCodeBlock {
        background: #1e1e2e !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(20, 20, 35, 0.95) !important;
    }
    
    /* Status Messages */
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
    
    /* Footer */
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
    
    /* Section Headers */
    .section-header {
        color: #00d2ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(40, 40, 60, 0.6);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric Cards */
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

# Sidebar for API key and settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    # Model selector
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
        
        # Show dataset info
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
        
        # Show column names
        st.markdown("**Columns:**")
        st.code(", ".join(df.columns.tolist()), language=None)

# Main content area
st.markdown(
    '<h1 class="main-title">💬 SheetTalk</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">✨ AI-Powered Chat with Your CSV Data ✨</p>',
    unsafe_allow_html=True
)

# Main chat interface
if st.session_state.api_key_set and st.session_state.df_uploaded:
    st.markdown("---")
    
    # Use tabs for different views
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Data"])
    
    with tab1:
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    if "code" in message:
                        st.markdown("**Generated Code:**")
                        st.code(message["code"], language="python")
                    if "result" in message:
                        st.markdown("**Result:**")
                        result = message["result"]
                        if isinstance(result, (pd.Series, pd.DataFrame)):
                            try:
                                st.dataframe(result)
                            except Exception:
                                st.write(
                                    result.to_dict()
                                    if isinstance(result, pd.Series)
                                    else result
                                )
                        else:
                            st.markdown(f"### {result}")
                    if "chart_data" in message:
                        st.markdown("**Visualization:**")
                        st.bar_chart(message["chart_data"])
        
        # Chat input at the bottom
        if prompt := st.chat_input("💭 Ask anything about your data..."):
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )
            
            df = st.session_state.df
            openai_prompt = f"""
You are an expert Python data analyst.

A pandas dataframe named df already exists.

Columns in df:
{list(df.columns)}

Your task:
Convert the user's question into a SINGLE executable pandas command.

STRICT RULES:
- Use ONLY the dataframe df
- Use pandas operations only
- Do NOT create variables
- Do NOT use print()
- Return ONE line of valid Python code
- The code must directly return the result

Examples:

Question: total revenue by city
Answer:
df.groupby('city')['revenue'].sum()

Question: city with highest revenue
Answer:
df.groupby('city')['revenue'].sum().idxmax()

Now answer:

Question: {prompt}
"""
            
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    client = st.session_state.client
                    completion = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {"role": "user", "content": openai_prompt}
                        ],
                        temperature=0
                    )
                    
                    code = completion.choices[0].message.content
                    
                    # Clean markdown formatting
                    code = (
                        code.replace("```python", "")
                        .replace("```", "")
                        .strip()
                    )
                    
                    # Extract the line containing df operation
                    lines = code.split("\n")
                    code = [line for line in lines if "df" in line][0]
                    
                    # Execute generated code
                    result = eval(code, {"df": df, "pd": pd})
                    
                    response_data = {
                        "role": "assistant",
                        "code": code,
                        "result": result
                    }
                    
                    if isinstance(result, (pd.Series, pd.DataFrame)):
                        response_data["chart_data"] = result
                    
                    st.session_state.chat_history.append(response_data)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
            
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
        try:
            st.dataframe(df.head(10))
        except Exception:
            st.write(df.head(10))
        
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': [str(dtype) for dtype in df.dtypes.values],
            'Non-Null': df.notna().sum().values,
            'Null': df.isna().sum().values
        })
        try:
            st.dataframe(col_info)
        except Exception:
            st.write(col_info)

elif not st.session_state.api_key_set:
    welcome_msg = """
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
             color: #e0e0e0; padding: 20px; border-radius: 20px;
             border: 1px solid rgba(255, 255, 255, 0.1);">
            👋 Welcome to <b>SheetTalk</b>!<br><br>
            I'm your AI-powered data analyst. To get started:<br>
            1. Enter your OpenAI API key in the sidebar<br>
            2. Upload a CSV file<br>
            3. Start asking questions about your data!<br><br>
            💡 Get your API key at
            <a href="https://platform.openai.com/api-keys" target="_blank">
            platform.openai.com</a>
        </div>
    </div>
    """
    st.markdown(welcome_msg, unsafe_allow_html=True)
    
    st.markdown("---")
    features_html = """
    <div style="display: flex; justify-content: space-around;
         text-align: center; color: #a0a0a0;">
        <div>
            <div style="font-size: 2rem;">📊</div>
            <div>Upload CSV</div>
        </div>
        <div>
            <div style="font-size: 2rem;">💬</div>
            <div>Ask Questions</div>
        </div>
        <div>
            <div style="font-size: 2rem;">🤖</div>
            <div>AI Analysis</div>
        </div>
        <div>
            <div style="font-size: 2rem;">📈</div>
            <div>Visualize</div>
        </div>
    </div>
    """
    st.markdown(features_html, unsafe_allow_html=True)

elif not st.session_state.df_uploaded:
    upload_msg = """
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
             color: #e0e0e0; padding: 20px; border-radius: 20px;
             border: 1px solid rgba(255, 255, 255, 0.1);">
            👋 API Key connected! Now please upload a CSV file to continue.<br><br>
            💡 You can use the file uploader in the sidebar to upload your data.
        </div>
    </div>
    """
    st.markdown(upload_msg, unsafe_allow_html=True)

# Footer
st.markdown("---")
year = pd.Timestamp.now().year
footer_html = f"""
<div class="footer">
    © {year} SheetTalk. All rights reserved.<br>
    Made with ❤️ for data enthusiasts | Powered by <span>Dhokla</span>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)