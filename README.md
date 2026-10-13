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
🧠 Conceptual Reference: 
Semantic vs. Keyword Search Feature: Keyword-Based Search (e.g., BM25, TF-IDF)Semantic Search (ChromaDB + Sentence-Transformers)Matching Mechanism Exact literal string and token frequency matching.Dense vector distance proximity in high-dimensional space.
Handling Synonyms: Fails if search terms do not match document text (e.g., searching "car" misses "automobile").
Captures underlying conceptual intent and groups synonyms close together. 
Context Awareness: Treats words independently or via basic n-gram frequencies.
Evaluates sentence-level contextual semantics and multi-word meaning.

## 🚀 Setup & Execution Guide
Prerequisites
Python 3.10 or higher installed on your system.

Git installed for version control.

### 1. Clone & Set Up Environment
```bash
git clone <your-repository-url>
cd qa_system
python -m venv venv
