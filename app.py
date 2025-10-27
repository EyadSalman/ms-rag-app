import streamlit as st
import requests
import plotly.express as px
from datetime import datetime
from supabase_client import supabase
import os
import uuid
import time
import streamlit.components.v1 as components
import pandas as pd


# ==============================
# ⚙️ CONFIG
# ==============================
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="DetectMS Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 MODERN CSS STYLING (TAILWIND-INSPIRED)
# ==============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Google Button */
    .google-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.875rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        background: white;
        color: #2d3748;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: 1rem;
        width: 100%;
        text-decoration: none;
    }
    
    .google-btn:hover {
        background: #f7fafc;
        border-color: #cbd5e0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .google-logo {
        width: 24px;
        height: 24px;
    }
    
    /* Chat Sidebar Button */
    .chat-history-btn {
        width: 100%;
        text-align: left;
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .chat-history-btn:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
    
     /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid;
    }       
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 🧠 SESSION STATE INIT
# ==============================
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "efficientnet"
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "login"
if "trigger_tab_switch" not in st.session_state:
    st.session_state.trigger_tab_switch = 0

# Refresh Supabase session if token present in browser cookies
session = supabase.auth.get_session()
if session and session.user:
    st.session_state.user = {
        "email": session.user.email,
        "id": session.user.id,
    }

# ============================================================
# 🎨 JAVASCRIPT TO SWITCH TO CHATBOT TAB
# ============================================================
def switch_to_chatbot_tab():
    """Inject JavaScript to switch to the chatbot tab"""
    trigger_id = st.session_state.get("trigger_tab_switch", 0)
    
    components.html(
        f"""
        <script>
            // Unique trigger ID: {trigger_id}
            function clickChatbotTab_{trigger_id}() {{
                const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                console.log('Trigger {trigger_id}: Found tabs:', tabs.length);
                
                if (tabs && tabs.length >= 3) {{
                    tabs[2].click();
                    console.log('Trigger {trigger_id}: Clicked chatbot tab');
                }}
            }}
            
            clickChatbotTab_{trigger_id}();
            setTimeout(clickChatbotTab_{trigger_id}, 50);
            setTimeout(clickChatbotTab_{trigger_id}, 100);
            setTimeout(clickChatbotTab_{trigger_id}, 200);
            setTimeout(clickChatbotTab_{trigger_id}, 300);
        </script>
        """,
        height=0
    )
    
    st.session_state.trigger_tab_switch = 0

# ============================================================
# 📜 Load previous chat history from Supabase
# ============================================================
def load_chat_history():
    """Load all previous chat sessions for the user from Supabase."""
    try:
        user = st.session_state.get("user")
        if not user:
            return

        if "chat_sessions" in st.session_state and st.session_state.chat_sessions:
            return

        rows = (
            supabase.table("chat_history")
            .select("session_id, query, response, created_at")
            .eq("user_id", user["id"])
            .order("created_at", desc=False)
            .execute()
            .data
        )

        if not rows:
            return

        sessions = {}
        for row in rows:
            sid = row["session_id"]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append({"role": "user", "content": row["query"]})
            sessions[sid].append({"role": "assistant", "content": row["response"]})

        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
        st.session_state.chat_sessions.update(sessions)

        if not st.session_state.active_chat and sessions:
            st.session_state.active_chat = list(sessions.keys())[-1]

        if st.session_state.active_chat:
            st.session_state.messages = st.session_state.chat_sessions[st.session_state.active_chat]
        
        st.success("💬 Chat history loaded successfully.")
    except Exception as e:
        st.warning(f"⚠️ Could not load chat history: {e}")


# =====================================================
# 🔑 LOGIN COMPONENT
# =====================================================
def login_component():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    if "user" not in st.session_state:
        st.session_state.user = None

    # Handle OAuth redirect
    if "code" in st.query_params:
        code = st.query_params["code"]
        st.info("🎟️ Exchanging authorization code for session...")

        try:
            data = supabase.auth.exchange_code_for_session({"auth_code": code})
            if data and data.user:
                st.session_state.user = {
                    "email": data.user.email,
                    "id": data.user.id,
                }
                st.success(f"✅ Logged in as {data.user.email}")
                load_chat_history()
                st.query_params.clear()
                time.sleep(0.8)
                st.rerun()
                return
        except Exception as e:
            st.error(f"❌ Failed to exchange code: {e}")
            return

    # Check for existing session
    session = supabase.auth.get_session()
    if session and session.user:
        # ✅ Logged-in state (hide login message)
        st.session_state.user = {
            "email": session.user.email,
            "id": session.user.id,
        }
        st.success(f"✅ Logged in as {session.user.email}")
        st.markdown(f'<div class="badge badge-success">Active: {session.user.email}</div>', unsafe_allow_html=True)
        load_chat_history()

        if st.button("🚪 Logout", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.session_state.chat_sessions = {}
            st.session_state.messages = []
            st.session_state.active_chat = None
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # =====================================================
    # 👇 Only show this if user is NOT logged in
    # =====================================================
    st.markdown('<p style="color: #718096; margin-bottom: 2rem;">Sign in to access your DetectMS Assistant</p>', unsafe_allow_html=True)

    # Show "Sign in with Google" button
    redirect_url = "http://localhost:8501"

    auth = supabase.auth.sign_in_with_oauth(
        {"provider": "google", "options": {"redirect_to": redirect_url}}
    )

    st.markdown(f"""
        <a href="{auth.url}" target="_self" style="text-decoration: none;">
            <div class="google-btn">
                <img src="https://www.svgrepo.com/show/355037/google.svg" class="google-logo"/>
                <span>Sign in with Google</span>
            </div>
        </a>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 📊 DASHBOARD COMPONENT
# =====================================================
def dashboard_component():
    st.markdown('<h1 class="gradient-header">MRI Analysis Dashboard</h1>', unsafe_allow_html=True)

    # ✅ Check user login
    user = st.session_state.get("user")
    if not user or not user.get("id"):
        st.warning("⚠️ Please login first to access the dashboard.")
        return

    # =====================================================
    # 📤 Upload and Analyze MRI
    # =====================================================
    with st.expander("📤 Upload and Analyze New MRI Scan", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            model_choice = st.radio(
                "🧠 Choose AI Model:",
                ["mobilenet", "efficientnet", "cnn", "vit", "densenet", "resnet"],
                horizontal=True,
                key="model_choice",
            )

        with col2:
            st.markdown(
                f'<div class="badge badge-info">Selected: {model_choice.upper()}</div>',
                unsafe_allow_html=True
            )

        uploaded = st.file_uploader("📁 Upload an MRI image", type=["png", "jpg", "jpeg"], key="mri_file")

        if uploaded:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded, caption="📸 Uploaded MRI", use_container_width=True)

            with col2:
                if st.button("🔍 Analyze MRI", key="analyze_btn", use_container_width=True):
                    with st.spinner(f"Analyzing MRI using {model_choice.upper()}..."):
                        try:
                            files = {"file": uploaded.getvalue()}
                            r = requests.post(
                                f"{FASTAPI_URL}/predict/{model_choice}", files=files, timeout=60
                            )

                            if r.status_code == 200:
                                res = r.json()
                                color = "🟢" if "Healthy" in res["diagnosis"] else "🔴"

                                st.markdown("### Analysis Results")
                                st.metric(
                                    f"{color} {res['model'].upper()} Diagnosis",
                                    res["diagnosis"],
                                    f"Confidence: {res['confidence']}%",
                                )

                                # ✅ Save results to Supabase
                                supabase.table("mri_results").insert({
                                    "user_id": user["id"],
                                    "diagnosis": res["diagnosis"],
                                    "confidence": res["confidence"],
                                    "created_at": datetime.now().isoformat()
                                }).execute()
                                st.success("✅ Results saved to your history!")
                            else:
                                st.error(f"❌ Error {r.status_code}: {r.text}")
                        except Exception as e:
                            st.error(f"❌ Backend error: {e}")

    # =====================================================
    # 🩻 MRI HISTORY SECTION
    # =====================================================
    st.markdown("---")
    st.markdown('<h2 class="gradient-header">🩻 Your MRI Scan History</h2>', unsafe_allow_html=True)

    try:
        response = (
            supabase.table("mri_results")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .execute()
        )

        results = response.data if response and response.data else []

        if not results:
            st.info("📭 No MRI results yet. Upload a scan above to get started.")
            return

        df = pd.DataFrame(results)

        # Remove unwanted columns if present
        drop_cols = [c for c in ["id", "image_url", "user_id"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        # 📈 Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Scans", len(df))
        with col2:
            healthy_count = sum("Healthy" in d for d in df["diagnosis"])
            st.metric("🟢 Healthy", healthy_count)
        with col3:
            ms_count = len(df) - healthy_count
            st.metric("🔴 MS Detected", ms_count)


        # Render each MRI result with delete button
        for record in results:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 0.5])

                with col1:
                    st.text(f"User ID: {record['user_id']}")    
                with col2:
                    st.text(f"Diagnosis: {record['diagnosis']}")
                with col3:
                    st.text(f"Confidence: {record['confidence']}%")
                with col4:
                    st.text(f"Date: {record['created_at'][:19].replace('T', ' ')}")
                with col4:
                    if st.button("🗑️", key=f"del_{record['id']}"):
                        try:
                            supabase.table("mri_results").delete().eq("id", record["id"]).execute()
                            st.success("✅ Deleted successfully!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"⚠️ Error deleting record: {e}")

                st.markdown(
                    "<hr style='margin-top:0.5rem;margin-bottom:0.5rem;border:1px solid #2d2d2d;'>",
                    unsafe_allow_html=True
                )

        # 📉 Chart visualization
        fig = px.bar(
            df,
            x="created_at",
            y="confidence",
            color="diagnosis",
            title="📈 MRI Scan Confidence Over Time",
            labels={"confidence": "Confidence (%)", "created_at": "Date"},
            color_discrete_map={
                "Multiple Sclerosis Detected": "#f56565",
                "Healthy Brain": "#48bb78",
                "Healthy": "#48bb78"
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")


# =====================================================
# 🗑️ Helper: Delete Chat Session
# =====================================================
def delete_chat_session(session_id: str):
    """Delete a single chat session from both local state and Supabase."""
    try:
        user = st.session_state.get("user")
        if not user:
            return

        if session_id in st.session_state.chat_sessions:
            del st.session_state.chat_sessions[session_id]

        if st.session_state.active_chat == session_id:
            st.session_state.active_chat = None
            st.session_state.messages = []

        supabase.table("chat_history") \
            .delete() \
            .eq("user_id", user["id"]) \
            .eq("session_id", session_id) \
            .execute()
    except Exception as e:
        st.error(f"⚠️ Failed to delete chat: {e}")


# =====================================================
# 💬 CHATBOT COMPONENT
# =====================================================
def chatbot_component():
    st.markdown('<h1 class="gradient-header">AI Health Assistant</h1>', unsafe_allow_html=True)

    if not st.session_state.user:
        st.warning("⚠️ Please login first to use the chatbot.")
        return

    # Sidebar — Chat History List
    with st.sidebar:
        st.markdown("### 💬 Chat History")

        if st.session_state.chat_sessions:
            for sid, msgs in list(st.session_state.chat_sessions.items())[::-1]:
                cols = st.columns([4, 1])
                label = msgs[0]["content"][:40] + "..." if msgs else f"Chat {sid[:5]}"

                with cols[0]:
                    if st.button(label, key=f"chat_{sid}"):
                        st.session_state.active_chat = sid
                        st.session_state.messages = msgs
                        st.rerun()

                with cols[1]:
                    if st.button("🗑️", key=f"delete_{sid}"):
                        delete_chat_session(sid)
                        st.rerun()
        else:
            st.info("📭 No previous chats yet.")

        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
            new_sid = str(uuid.uuid4())
            st.session_state.chat_sessions[new_sid] = []
            st.session_state.active_chat = new_sid
            st.session_state.messages = []
            st.rerun()

    # Main Chat Area
    if st.session_state.active_chat is None:
        if st.button("🚀 Start New Chat", key="start_chat_in_main", use_container_width=True):
            new_sid = str(uuid.uuid4())
            st.session_state.chat_sessions[new_sid] = []
            st.session_state.active_chat = new_sid
            st.session_state.messages = []
            st.rerun()
        return

    # Render chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("💭 Ask about Multiple Sclerosis..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.spinner("Generating response..."):
                # 👇 CRITICAL: Pass the active_chat session_id to backend
                res = requests.post(
                    f"{FASTAPI_URL}/ask_gemini/",
                    json={
                        "query": prompt,
                        "history": st.session_state.messages,
                        "user_id": st.session_state.user["id"],
                        "session_id": st.session_state.active_chat,  # 👈 This is the key!
                    },
                    timeout=60,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response.")
                    st.chat_message("assistant").markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # 👇 Update the session in memory
                    sid = st.session_state.active_chat
                    st.session_state.chat_sessions[sid] = st.session_state.messages
                    st.rerun()
                else:
                    st.error(f"❌ Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"❌ Backend error: {e}")


# =====================================================
# 🧭 MAIN APP LAYOUT
# =====================================================
st.markdown('<h1 class="gradient-header" style="text-align: center;">🏥 DetectMS Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #4a5568; font-size: 1.2rem; margin-bottom: 2rem;">AI-Powered Multiple Sclerosis Detection & Support</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔑 Login", "📊 Dashboard", "💬 Chatbot"])

with tab1:
    login_component()

with tab2:
    dashboard_component()

with tab3:
    chatbot_component()

# Trigger tab switch if needed
if st.session_state.trigger_tab_switch > 0:
    switch_to_chatbot_tab()