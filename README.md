# AI-Powered Document Question Answering System

An intelligent study assistant application built with **Streamlit**, **Sentence-Transformers**, and **ChromaDB**. It allows users to perform Prompt Engineering analysis and query uploaded documents using vector embeddings and semantic search.

---

## 📌 Features

1. **Part 1: Prompt Engineering**
   - Compares **Zero-Shot**, **Few-Shot**, and **Role-Based** prompting techniques.
   - Tests prompts using identical context and question inputs.
   - Provides written performance evaluation and comparison.

2. **Part 2: Embeddings & ChromaDB Semantic Search**
   - Ingests multiple PDF, TXT, and Markdown documents.
   - Chunks text into smaller segments and generates dense embeddings using `all-MiniLM-L6-v2`.
   - Stores vectors in **ChromaDB** with Cosine similarity space.
   - Retrieves top 3 semantically relevant chunks with distance and similarity scores.
   - Explains differences between Semantic Search and Keyword-Based Search.

---

## 🚀 Setup & Execution Guide

### 1. Clone & Set Up Environment
```bash
git clone <your-repository-url>
cd qa_system
python -m venv venv