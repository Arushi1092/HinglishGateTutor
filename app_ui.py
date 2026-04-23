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
                    st.success("Indexing complete!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

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
