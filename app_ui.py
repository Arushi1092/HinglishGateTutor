import streamlit as st
import requests
import time
import json

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
    
    # Tabs for different functions
    tab1, tab2 = st.tabs(["📁 Ingestion", "🧪 Quality"])

    with tab1:
        # Check Backend Status
        try:
            health_resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if health_resp.status_code == 200:
                data = health_resp.json()
                st.success(f"Backend: Online ({data.get('chunks_indexed', 0)} chunks)")
            else:
                st.warning("Backend: Issues detected")
        except:
            st.error("Backend: Offline")

        st.divider()

        # File Ingestion
        st.subheader("📁 Add PDF")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.button("Ingest PDF"):
            with st.spinner("Processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{BASE_URL}/ingest", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Indexed {data.get('ingested_chunks', 0)} chunks!")
                        st.rerun()
                    else:
                        st.error(f"Failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        # Wikipedia Ingestion
        st.subheader("🌐 Wikipedia")
        wiki_title = st.text_input("Page Title", placeholder="e.g. Calculus")
        if st.button("Fetch & Ingest"):
            if wiki_title:
                with st.spinner(f"Fetching '{wiki_title}'..."):
                    try:
                        resp = requests.post(f"{BASE_URL}/wiki", json={"title": wiki_title})
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"Indexed {data.get('ingested_chunks', 0)} chunks!")
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()
        st.subheader("📚 Library")
        try:
            sources_resp = requests.get(f"{BASE_URL}/sources")
            if sources_resp.status_code == 200:
                sources = sources_resp.json().get("sources", [])
                for src in sources:
                    col1, col2 = st.columns([0.8, 0.2])
                    col1.text(f"📄 {src[:15]}...")
                    if col2.button("🗑️", key=f"del_{src}"):
                        requests.delete(f"{BASE_URL}/source/{src}")
                        st.rerun()
        except:
            st.caption("No sources found.")

    with tab2:
        st.subheader("🧪 Golden Dataset")
        try:
            golden_resp = requests.get(f"{BASE_URL}/golden_qa")
            if golden_resp.status_code == 200:
                golden_data = golden_resp.json()
                st.info(f"Loaded {len(golden_data)} test cases.")
                for item in golden_data:
                    with st.expander(item["question"]):
                        st.caption("Golden Answer:")
                        st.write(item["golden_answer"])
                        if st.button("Run Test", key=f"test_{item['question']}"):
                            # We can trigger a specific test here
                            pass
            else:
                st.error("Failed to load golden data.")
        except:
            st.error("Backend not reachable.")
        
        st.divider()
        st.subheader("⚙️ Reset")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

# --- MAIN INTERFACE ---
st.title("Hindi-GATE Tutor Chat")
st.markdown("Ask questions about your uploaded syllabus PDFs in English or Hindi.")

# Helper to display evaluation
def display_evaluation(eval_data):
    if eval_data:
        score = eval_data.get("accuracy", 0)
        reasoning = eval_data.get("reasoning", "")
        
        color = "green" if score >= 4 else "orange" if score >= 3 else "red"
        st.markdown(f"""
        <div style="border:1px solid {color}; padding:10px; border-radius:5px; margin-bottom:10px;">
            <b style="color:{color};">Knowledge Quality Score: {score}/5</b><br>
            <small>{reasoning}</small>
        </div>
        """, unsafe_allow_html=True)

# Display Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "evaluation" in msg and msg["evaluation"]:
            display_evaluation(msg["evaluation"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "history": st.session_state.history, "top_k": 5}
                resp = requests.post(f"{BASE_URL}/ask", json=payload)
                
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    evaluation = data.get("evaluation")
                    
                    st.markdown(answer)
                    if evaluation:
                        display_evaluation(evaluation)
                    
                    if sources:
                        st.caption(f"Sources: {', '.join(sources)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources,
                        "evaluation": evaluation
                    })
                    st.session_state.history.append({"q": prompt, "a": answer})
                else:
                    st.error(f"Backend Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")
