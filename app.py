import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

st.set_page_config(page_title="PDF AI Chatbot", page_icon="🤖")

st.title("🤖 Domain-Specific RAG Chatbot")
st.write("Upload PDFs and ask questions about their contents.")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        all_chunks = []
        all_sources = []

        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=120
                    )

                    chunks = splitter.split_text(text)

                    for chunk in chunks:
                        all_chunks.append(chunk)
                        all_sources.append(
                            f"{uploaded_file.name}, page {page_number}"
                        )

        if all_chunks:
            embeddings = model.encode(
                all_chunks,
                normalize_embeddings=True
            )

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(np.asarray(embeddings, dtype="float32"))

            st.session_state["chunks"] = all_chunks
            st.session_state["sources"] = all_sources
            st.session_state["index"] = index

            st.session_state["chat_history"] = []

            st.success("PDF documents processed successfully!")
        else:
            st.warning("No readable text found in the uploaded PDFs.")
    else:
        st.warning("Please upload at least one PDF.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))

question = st.chat_input("Ask a question about your PDF")

if question:
    st.session_state["chat_history"].append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if "index" not in st.session_state:
            answer = "Please upload and process a PDF first."
            st.warning(answer)

        else:
            query_embedding = model.encode(
                [question],
                normalize_embeddings=True
            )

            scores, indices = st.session_state["index"].search(
                np.asarray(query_embedding, dtype="float32"),
                min(4, len(st.session_state["chunks"]))
            )

            relevant_chunks = []
            source_list = []

            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.25:
                    relevant_chunks.append(
                        st.session_state["chunks"][idx]
                    )

                    source = st.session_state["sources"][idx]

                    if source not in source_list:
                        source_list.append(source)

            if not relevant_chunks:
                answer = "The answer was not found in the uploaded PDF."
                st.write(answer)

            else:
                context = "\n\n".join(relevant_chunks)

                prompt = f"""
You are a helpful PDF question-answering assistant.

Rules:
1. Answer using only the provided context.
2. Give a clear, direct answer in simple language.
3. Do not repeat the context word for word unless necessary.
4. If the context does not contain the answer, say:
   "The answer was not found in the uploaded PDF."
5. Do not invent facts.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2:1b",
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=120
                    )

                    response.raise_for_status()
                    answer = response.json()["response"].strip()

                    if not answer:
                        answer = (
                            "The answer was not found in the uploaded PDF."
                        )

                    st.write(answer)

                    st.caption(
                        "Sources: " + "; ".join(source_list)
                    )

                except requests.RequestException:
                    answer = (
                        "Could not connect to Ollama. "
                        "Please check that Ollama is running."
                    )
                    st.error(answer)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "sources": source_list if (
                "index" in st.session_state and relevant_chunks
            ) else []
        })