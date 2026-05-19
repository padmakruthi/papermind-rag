"""
PaperMind — RAG for Scientific Papers
Upload PDF → Extract → Chunk → Embeddings → FAISS → Ask Questions
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import build_rag_pipeline, ask_question

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="PaperMind · Scientific Paper Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────
st.markdown("""
<style>
body {background-color:#0d0f14;}
.hero-title{
font-size:2.5rem;
font-weight:700;
color:#00e5c8;
}
.answer-box{
background:#131720;
color:#ffffff;   /* FIX: make text white */
padding:1.2rem;
border-radius:8px;
border-left:4px solid #00e5c8;
font-size:16px;
line-height:1.6;
}
.source-box{
background:#0f1320;
padding:0.8rem;
border-radius:6px;
margin-top:6px;
font-size:0.85rem;
color:#9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "paper_name" not in st.session_state:
    st.session_state.paper_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Layout ──────────────────────────────────────────────
col1, col2 = st.columns([1,2])

# ── LEFT PANEL ──────────────────────────────────────────
with col1:

    st.markdown('<div class="hero-title">🔬 PaperMind</div>', unsafe_allow_html=True)
    st.write("AI assistant for reading research papers.")

    st.markdown("### Upload Research Paper")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("⚡ Build Knowledge Index"):

        if uploaded_file is None:
            st.error("Upload a research paper first.")

        else:

            with st.spinner("Processing paper..."):

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    pdf_path = tmp.name

                try:

                    st.session_state.qa_chain = build_rag_pipeline(pdf_path)
                    st.session_state.paper_name = uploaded_file.name
                    st.session_state.messages = [] # Clear previous chat messages

                    os.unlink(pdf_path)

                    st.success("Paper processed successfully!")

                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.paper_name:
        st.info(f"Active paper: {st.session_state.paper_name}")


# ── RIGHT PANEL ─────────────────────────────────────────
with col2:

    st.markdown("### Chat with Paper")

    # Display chat messages
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("View Sources"):
                        for doc in msg["sources"]:
                            st.markdown(f'<div class="source-box">{doc}</div>', unsafe_allow_html=True)

    # Chat input
    if question := st.chat_input("Ask a question about the paper..."):

        if st.session_state.qa_chain is None:
            st.error("Upload and process a paper first.")

        else:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": question})

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Searching paper and generating answer..."):
                        try:
                            result = ask_question(st.session_state.qa_chain, question)
                            answer = result["answer"]

                            sources = []
                            for doc in result["source_documents"]:
                                page = doc.metadata.get("page", "Unknown")
                                snippet = doc.page_content[:350].replace("\n", " ")
                                sources.append(f"**Page {page}**: {snippet}...")

                            st.markdown(answer)
                            with st.expander("View Sources"):
                                for doc in sources:
                                    st.markdown(f'<div class="source-box">{doc}</div>', unsafe_allow_html=True)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })

                        except Exception as e:
                            st.error(f"Error generating answer: {e}")