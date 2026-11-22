import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# --- HELPERS ---
def render_steps(steps):
    """Draw the thought process once, inside whatever container is active."""
    for step in steps:
        st.write(f"⚙️ {step}")


def render_sources(sources):
    """Draw retrieved chunks inside a single expander."""
    with st.expander("📄 View Retrieved Context (Sources)"):
        for i, source in enumerate(sources):
            preview = str(source)[:100].replace("\n", " ") + "..."
            with st.expander(f"Chunk {i+1}: {preview}"):
                st.info(source)


# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")

    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")


# Display history (old messages only; the new turn is drawn live below)
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Replay the thought process and sources saved with the answer
        if message.get("steps"):
            with st.expander("⚙️ Thought process"):
                render_steps(message["steps"])
        if message.get("sources"):
            render_sources(message["sources"])


# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            # Defined up front so they always exist, even if something fails
            data = {}
            steps = []
            sources = []

            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        payload = {"q": prompt, "thread_id": st.session_state.session_id}
                        response = requests.post(url, json=payload, timeout=60)
                        response.raise_for_status()
                        data = response.json()

                    # Show Reasoning Steps from Backend (drawn once)
                    steps = data.get("thought_process") or []
                    render_steps(steps)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # --- SHOW SOURCES (outside the status box so they stay visible) ---
            sources = data.get("sources") or []
            if sources:
                render_sources(sources)

            # Final Answer Streaming (the ONLY place the answer is drawn)
            answer_placeholder = st.empty()
            full_answer = data.get("answer") or "No response generated."

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)

            answer_placeholder.markdown(full_answer)

            # Save AFTER drawing, with steps and sources so history can replay them
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "steps": steps,
                "sources": sources,
            })
            logfire.info("✅ Chat cycle completed successfully.")