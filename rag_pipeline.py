"""
RAG Pipeline for Scientific Papers
Local embeddings + FAISS + HuggingFace LLM
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline


# ── Build RAG Pipeline ─────────────────────────
def build_rag_pipeline(pdf_path):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(pages)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Local LLM
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256,
        device=-1
    )

    llm = HuggingFacePipeline(pipeline=generator)

    # Prompt Template
    prompt_template = """
Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Give a short clear answer.
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain


# ── Ask Question ─────────────────────────
def ask_question(qa_chain, question):

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }