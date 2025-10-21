import streamlit as st
import requests
import plotly.express as px
from datetime import datetime
from supabase_client import supabase
import os
import uuid
import re
import streamlit as st
from supabase_client import supabase
import time
import urllib.parse

# ==============================
# ⚙️ CONFIG
# ==============================
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="🧠 MS Detection Assistant", layout="wide")

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
    st.session_state.chat_sessions = {}  # {session_id: [{"role": "user/assistant", "content": "..."}]}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

# Refresh Supabase session if token present in browser cookies
session = supabase.auth.get_session()
if session and session.user:
    st.session_state.user = {
        "email": session.user.email,
        "id": session.user.id,
    }


# =====================================================
# 🔑 LOGIN COMPONENT
# =====================================================

def login_component():
    st.subheader("🔑 Login to MS Detection Assistant")

    if "user" not in st.session_state:
        st.session_state.user = None

    # 🧩 STEP 1: Handle OAuth redirect (?code=...)
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

                # Clean up query params (remove ?code=...)
                st.query_params.clear()
                time.sleep(0.8)
                st.rerun()
                return
        except Exception as e:
            st.error(f"❌ Failed to exchange code: {e}")
            return

    # 🧩 STEP 2: Check for an existing session
    session = supabase.auth.get_session()
    if session and session.user:
        st.session_state.user = {
            "email": session.user.email,
            "id": session.user.id,
        }
        st.success(f"✅ Logged in as {session.user.email}")

        if st.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()
        return

    # 🧩 STEP 3: Show "Sign in with Google" button
    redirect_url = "http://localhost:8501"

    st.markdown("""
        <style>
        .google-btn {
            display:flex;align-items:center;justify-content:center;
            gap:10px;border:1px solid #ccc;border-radius:6px;
            padding:10px 16px;font-size:16px;background-color:white;
            cursor:pointer;transition:background-color 0.2s;
        }
        .google-btn:hover {background-color:#f7f7f7;}
        .google-logo {width:20px;height:20px;}
        </style>
    """, unsafe_allow_html=True)

    if st.button("Sign in with Google"):
        auth = supabase.auth.sign_in_with_oauth(
            {"provider": "google", "options": {"redirect_to": redirect_url}}
        )
        st.markdown(
            f'<meta http-equiv="refresh" content="0; url={auth.url}">',
            unsafe_allow_html=True,
        )

    # Optional pretty static button for layout
    st.markdown("""
        <div class="google-btn">
            <img src="https://www.svgrepo.com/show/355037/google.svg" class="google-logo"/>
            <span>Sign in with Google</span>
        </div>
    """, unsafe_allow_html=True)


# =====================================================
# 📊 DASHBOARD COMPONENT
# =====================================================
def dashboard_component():
    st.subheader("📊 MRI Analysis Dashboard")

    if not st.session_state.user:
        st.warning("Please login first.")
        return

    # ---- Upload new MRI ----
    with st.expander("📤 Upload and Analyze New MRI Scan", expanded=True):
        model_choice = st.radio(
            "Choose model for MRI analysis:",
            ["mobilenet", "efficientnet", "cnn", "vit", "densenet", "resnet"],
            horizontal=True,
            key="model_choice",
        )
        uploaded = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"], key="mri_file")

        if uploaded and st.button("Analyze MRI", key="analyze_btn"):
            with st.spinner(f"Analyzing MRI using {model_choice.upper()}..."):
                try:
                    files = {"file": uploaded.getvalue()}
                    r = requests.post(f"{FASTAPI_URL}/predict/{model_choice}", files=files, timeout=60)
                    if r.status_code == 200:
                        res = r.json()
                        color = "🟢" if "Healthy" in res["diagnosis"] else "🔴"
                        st.image(uploaded, caption="Uploaded MRI", use_container_width=True)
                        st.metric(
                            f"{color} {res['model'].upper()} Result",
                            res["diagnosis"],
                            f"{res['confidence']}%",
                        )
                        if st.session_state.user["id"]:
                            supabase.table("mri_results").insert({
                                "user_id": st.session_state.user["id"],
                                "diagnosis": res["diagnosis"],
                                "confidence": res["confidence"],
                                "created_at": datetime.now().isoformat()
                            }).execute()
                    else:
                        st.error(f"Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.error(f"Backend error: {e}")

    # ---- MRI upload history ----
    st.markdown("---")
    st.subheader("🩻 MRI Scan History")

    try:
        results = (
            supabase.table("mri_results")
            .select("*")
            .eq("user_id", st.session_state.user["id"])
            .order("created_at", desc=True)
            .execute()
            .data
        )

        if results:
            st.dataframe(results, use_container_width=True)
            fig = px.bar(
                results,
                x="created_at",
                y="confidence",
                color="diagnosis",
                title="MRI Scan Confidence Over Time",
                labels={"confidence": "Confidence (%)", "created_at": "Date"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No MRI results yet. Upload a scan above to get started.")
    except Exception as e:
        st.error(f"Dashboard error: {e}")

# =====================================================
# 💬 CHATBOT COMPONENT (with sidebar history)
# =====================================================
def chatbot_component():
    st.subheader("💬 Gemini Research Chatbot")

    if not st.session_state.user:
        st.warning("Please login first.")
        return

    # Sidebar — show past sessions
    with st.sidebar:
        st.markdown("### 💬 Chat History")

        if st.session_state.chat_sessions:
            for sid, msgs in list(st.session_state.chat_sessions.items())[::-1]:
                label = msgs[0]["content"][:40] + "..." if msgs else f"Chat {sid[:5]}"
                if st.button(label, key=f"chat_{sid}"):
                    st.session_state.active_chat = sid
                    st.session_state.messages = msgs
                    st.rerun()
        else:
            st.info("No previous chats yet.")

        if st.button("➕ New Chat", key="new_chat_btn"):
            new_sid = str(uuid.uuid4())
            st.session_state.chat_sessions[new_sid] = []
            st.session_state.active_chat = new_sid
            st.session_state.messages = []
            st.rerun()

    # Display current chat
    if st.session_state.active_chat is None:
        st.info("Start a new chat from the sidebar to begin.")
        return

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Ask about Multiple Sclerosis..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            with st.spinner("Gemini is thinking..."):
                res = requests.post(
                    f"{FASTAPI_URL}/ask_gemini/",
                    json={
                        "query": prompt,
                        "history": st.session_state.messages,
                        "user_id": st.session_state.user["id"],
                    },
                    timeout=60,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response.")
                    st.chat_message("assistant").markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    # ✅ Save in session-level chat history
                    sid = st.session_state.active_chat
                    st.session_state.chat_sessions[sid] = st.session_state.messages
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Backend error: {e}")

# =====================================================
# 🧭 NAVIGATION (TABS)
# =====================================================
st.title("🧠 MS Detection Assistant")

tab1, tab2, tab3 = st.tabs(["🔑 Login", "📊 Dashboard", "💬 Chatbot"])

with tab1:
    login_component()

with tab2:
    dashboard_component()

with tab3:
    chatbot_component()
