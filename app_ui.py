import streamlit as st
import requests
import time

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Hindi-GATE Tutor",
    page_icon="🎓",
    layout="wide"
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR (Status & Ingestion) ---
with st.sidebar:
    st.title("🎓 Control Panel")
    
    # Check Backend Status
    try:
        health_resp = requests.get(f"{BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success(f"Backend: Online ({data.get('chunks_indexed', 0)} chunks)")
        else:
            st.warning("Backend: Issues detected")
    except:
        st.error("Backend: Offline (Check Terminal)")
        st.info("Make sure to run: uvicorn app.main:app")

    st.divider()

    # File Ingestion
    st.subheader("📁 Add New Material")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Ingest PDF"):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                resp = requests.post(f"{BASE_URL}/ingest", files=files)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "already_ingested":
                        st.info(f"'{uploaded_file.name}' is already in the knowledge base.")
                    else:
                        st.success(f"Successfully indexed {data.get('ingested_chunks', 0)} chunks!")
                        time.sleep(2)
                        st.rerun()
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    
    # Document Library
    st.subheader("📚 Your Library")
    try:
        sources_resp = requests.get(f"{BASE_URL}/sources")
        if sources_resp.status_code == 200:
            sources = sources_resp.json().get("sources", [])
            if not sources:
                st.caption("No documents indexed yet.")
            for src in sources:
                col1, col2 = st.columns([0.8, 0.2])
                col1.text(f"📄 {src[:20]}...")
                if col2.button("🗑️", key=f"del_{src}"):
                    with st.spinner("Deleting..."):
                        requests.delete(f"{BASE_URL}/source/{src}")
                        st.rerun()
        else:
            st.error("Failed to load library.")
    except:
        st.caption("Connect to backend to see library.")

    st.divider()
    
    # Database Management
    st.subheader("⚙️ Management")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    if st.button("⚠️ Reset Database"):
        if st.checkbox("Confirm: Delete ALL indexed data?"):
            try:
                # We'll need to add a /reset endpoint to the backend
                resp = requests.post(f"{BASE_URL}/reset")
                if resp.status_code == 200:
                    st.success("Database cleared successfully!")
                    st.session_state.messages = []
                    st.session_state.history = []
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to reset database.")
            except:
                st.error("Reset endpoint not found in backend.")

# --- MAIN INTERFACE ---
st.title("Hindi-GATE Tutor Chat")
st.markdown("Ask questions about your uploaded syllabus PDFs in English or Hindi.")

# Display Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "history": st.session_state.history, "top_k": 5}
                resp = requests.post(f"{BASE_URL}/ask", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    st.session_state.history.append({"q": prompt, "a": answer})
                else:
                    st.error(f"Backend Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
