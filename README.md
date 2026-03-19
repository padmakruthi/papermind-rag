# 🔬 PaperMind — RAG for Scientific Papers

Query any research paper using natural language.
Built with **LangChain + OpenAI Embeddings + FAISS + Streamlit**.

## 📁 Folder Structure

```
rag-scientific-papers/
├── app.py              ← Streamlit frontend (UI)
├── rag_pipeline.py     ← RAG backend (PDF → chunks → embeddings → FAISS → LLM)
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## ⚙️ How It Works

```
PDF Upload
   ↓
PyPDFLoader  (extract text per page)
   ↓
RecursiveCharacterTextSplitter  (chunk_size=800, overlap=150)
   ↓
OpenAI Embeddings  (text-embedding-3-small)
   ↓
FAISS Vector Store  (in-memory index)
   ↓
User Question  →  Semantic Search (top-4 chunks)
   ↓
GPT-3.5-turbo  →  Final Answer + Source Chunks shown in UI


followsome steps
```

---


---

