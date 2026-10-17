# Domain-Specific RAG Chatbot for PDF Question Answering

## Project Overview
This project is an AI-powered chatbot that answers questions based on uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF documents
- Extract text from PDFs
- Split text into smaller chunks
- Generate embeddings using Sentence Transformers
- Retrieve relevant information using FAISS
- Generate answers using Ollama
- Display PDF filenames and page numbers as sources
- Provide a fallback when an answer is not found in the document

## Technologies Used
- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- NumPy
- Ollama

## Installation

1. Clone this repository.
2. Install the required packages:

   pip install -r requirements.txt

3. Install Ollama and download the model:

   ollama pull llama3.2:1b

4. Run the application:

   python -m streamlit run app.py

## How to Use
1. Start the application.
2. Upload one or more PDF files.
3. Click Process Documents.
4. Ask questions about the uploaded PDFs.
5. Read the AI-generated answer and its sources.

## Project Goal
To provide accurate, document-grounded answers from PDF files using Retrieval-Augmented Generation.