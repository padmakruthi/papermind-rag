# 🔬 PaperMind — RAG for Scientific Papers

Query any research paper using natural language.
Built with **LangChain + OpenAI Embeddings + FAISS + Streamlit**.

---

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
```

---

## 🚀 Run Locally

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/rag-scientific-papers.git
cd rag-scientific-papers
```

### 2. Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain langchain-community langchain-openai openai faiss-cpu pypdf streamlit
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### 5. Use the app

1. Paste your **OpenAI API key** (get one at https://platform.openai.com/api-keys)
2. Upload a PDF scientific paper
3. Click **"⚡ Build Knowledge Index"** — wait ~10–30 seconds
4. Type a question and click **"🔍 Search & Answer"**

---

## ☁️ Deploy to Streamlit Cloud (Free, Easiest)

Streamlit Cloud is the recommended free hosting — it natively supports Streamlit apps.

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/rag-scientific-papers.git
git push -u origin main
```

### 2. Deploy

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch (`main`), and main file (`app.py`)
5. Click **"Deploy"** — done! 🎉

> **Note on API key**: Do NOT hard-code your key. The app already accepts it at runtime via the UI text field, so no secrets config is needed.

---

## ☁️ Deploy to Vercel (Advanced)

> ⚠️ Vercel is designed for Node.js/Next.js apps. Streamlit does **not** run natively on Vercel.
> The workaround below wraps Streamlit in a Docker container deployed via Vercel's Container Runtime.

### Option A — Use Railway or Render instead (much easier for Python)

**Railway** (https://railway.app) and **Render** (https://render.com) both support Python/Streamlit with zero extra config:

```bash
# Railway
railway init
railway up
```

Set `STREAMLIT_SERVER_PORT=8501` in the Railway environment variables.

### Option B — Vercel with Dockerfile

#### Step 1 — Add a Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2 — Add vercel.json

```json
{
  "builds": [{ "src": "Dockerfile", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "/" }]
}
```

#### Step 3 — Deploy

```bash
npm i -g vercel
vercel --prod
```

> **Recommendation**: Use **Streamlit Cloud** or **Railway** for this app — they work without extra configuration and are free for small projects.

---

## 💡 Example Queries

Once a paper is indexed, try questions like:

| Question | What it tests |
|---|---|
| What is the main contribution of this paper? | Abstract-level understanding |
| What datasets were used for evaluation? | Methodology section |
| What are the limitations mentioned by the authors? | Discussion/conclusion |
| How does the proposed method compare to baselines? | Results tables |
| What future work do the authors suggest? | Conclusion section |

---

## 🔧 Customisation Tips

| Change | Where | How |
|---|---|---|
| Use GPT-4 | `rag_pipeline.py` | Change `"gpt-3.5-turbo"` → `"gpt-4o"` |
| Retrieve more chunks | `rag_pipeline.py` | Change `"k": 4` → `"k": 6` |
| Larger chunks | `rag_pipeline.py` | Increase `chunk_size=800` |
| Persist FAISS index | `rag_pipeline.py` | Add `vectorstore.save_local("faiss_index")` |

---

## 🧑‍🎓 Student Notes

This project demonstrates all core RAG concepts:

- **Retrieval**: FAISS cosine similarity search finds the most relevant chunks
- **Augmentation**: Retrieved chunks are injected into the LLM prompt as context
- **Generation**: GPT-3.5 reads the context and answers in natural language

Great for: AI/ML coursework, NLP projects, research tool prototypes.
