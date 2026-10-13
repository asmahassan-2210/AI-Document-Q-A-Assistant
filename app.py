import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os

# -------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Powered Document Question Answering System")
st.caption("A study assistant leveraging Prompt Engineering, Vector Embeddings, and ChromaDB Semantic Search.")

# -------------------------------------------------------------------
# SYSTEM INITIALIZATION (Cached for Speed)
# -------------------------------------------------------------------
@st.cache_resource
# -------------------------------------------------------------------
# SYSTEM INITIALIZATION (Cached for Speed)
# -------------------------------------------------------------------
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

model = get_embedding_model()
chroma_client = get_chroma_client()

# Dynamically obtain collection without storing stale references
def get_collection():
    return chroma_client.get_or_create_collection(
        name="study_assistant_docs",
        metadata={"hnsw:space": "cosine"}
    )

collection = get_collection()

# Safe collection reset inside the indexing block
try:
    chroma_client.delete_collection("study_assistant_docs")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name="study_assistant_docs",
    metadata={"hnsw:space": "cosine"}
)
# -------------------------------------------------------------------
# HELPER FUNCTIONS (Parsing & Text Chunking)
# -------------------------------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif filename.endswith((".txt", ".md")):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# -------------------------------------------------------------------
# SIDEBAR: DOCUMENT INGESTION
# -------------------------------------------------------------------
st.sidebar.header("📁 Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload Study Documents (PDF, TXT, MD)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) < 5:
        st.sidebar.warning(f"⚠️ Uploaded: {len(uploaded_files)} file(s). Assignment requires at least 5 files.")
    else:
        st.sidebar.success(f"✅ Ready: {len(uploaded_files)} files uploaded.")

    if st.sidebar.button("Process & Index Documents"):
        with st.spinner("Extracting text, generating embeddings, and storing in ChromaDB..."):
            all_chunks = []
            all_metadatas = []
            all_ids = []

            for file in uploaded_files:
                raw_text = extract_text_from_file(file)
                chunks = chunk_text(raw_text)
                for idx, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_metadatas.append({"filename": file.name, "chunk_index": idx})
                    all_ids.append(f"{file.name}_chunk_{idx}")

            if all_chunks:
                # Generate Embeddings
                embeddings = model.encode(all_chunks).tolist()
                
                # Reset collection and add new documents
                try:
                    chroma_client.delete_collection("study_assistant_docs")
                    collection = chroma_client.get_or_create_collection(
                        name="study_assistant_docs",
                        metadata={"hnsw:space": "cosine"}
                    )
                except Exception:
                    pass

                collection.add(
                    documents=all_chunks,
                    embeddings=embeddings,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
                st.sidebar.success(f"🚀 Indexed {len(all_chunks)} chunks across {len(uploaded_files)} files into ChromaDB!")

# -------------------------------------------------------------------
# MAIN NAVIGATION TABS
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📝 Part 1: Prompt Engineering", "🔍 Part 2: Embeddings & ChromaDB Search"])

# -------------------------------------------------------------------
# PART 1: PROMPT ENGINEERING COMPARISON
# -------------------------------------------------------------------
with tab1:
    st.header("Part 1: Prompt Engineering Techniques")
    st.markdown("Test and compare **Zero-Shot**, **Few-Shot**, and **Role-Based** prompt techniques using identical test inputs.")

    test_query = st.text_input(
        "Test Question:",
        value="What is Retrieval-Augmented Generation (RAG) and why is it useful?"
    )
    sample_context = st.text_area(
        "Context Document:",
        value="Retrieval-Augmented Generation (RAG) enhances Large Language Models by fetching factual information from external knowledge bases or vector databases before generating a response. This reduces hallucinations and ensures answers remain up-to-date."
    )

    if st.button("Generate & Compare Prompt Templates"):
        # Template 1: Zero-Shot
        p_zero = f"Answer the question based on the context.\n\nContext:\n{sample_context}\n\nQuestion:\n{test_query}"
        
        # Template 2: Few-Shot
        p_few = f"""Answer questions concisely based on context following the example format below.

Example 1:
Q: What is an embedding?
A: An embedding is a high-dimensional numerical vector representing semantic meaning.

Example 2:
Q: What is ChromaDB?
A: ChromaDB is an open-source vector database designed for efficient similarity searches.

Now Answer:
Context: {sample_context}
Question: {test_query}"""

        # Template 3: Role-Based
        p_role = f"""You are an Expert Technical Document Analyst and Academic Tutor. Your duty is to analyze the provided study context thoroughly and output a highly precise, structured, and fact-checked answer.

Context:
{sample_context}

Question:
{test_query}"""

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Zero-Shot Prompt")
            st.code(p_zero, language="markdown")
        with col2:
            st.subheader("2. Few-Shot Prompt")
            st.code(p_few, language="markdown")
        with col3:
            st.subheader("3. Role-Based Prompt")
            st.code(p_role, language="markdown")

    st.markdown("---")
    st.subheader("💡 Analysis & Comparison Evaluation")
    st.write("""
    * **Zero-Shot Prompting:** Delivers direct answers without sample demonstrations. Works well for simple requests, but tone, length, and depth can fluctuate.
    * **Few-Shot Prompting:** Guides response structure, output format, and desired brevity through explicit pattern examples.
    * **Role-Based Prompting (Best Performer):** Assigning an explicit system persona (*Technical Document Analyst*) forces strict boundary compliance, reduces hallucinations, and delivers high domain-specific rigor.
    """)

# -------------------------------------------------------------------
# PART 2: EMBEDDINGS & CHROMADB SEMANTIC SEARCH
# -------------------------------------------------------------------
with tab2:
    st.header("Part 2: Embeddings & ChromaDB Semantic Search")
    
    if collection.count() == 0:
        st.info("👈 Please upload and process at least 5 study documents using the sidebar to populate ChromaDB.")
    else:
        user_query = st.text_input("Ask a question about your uploaded documents:")
        
        if user_query:
            # Generate query vector
            query_vector = model.encode([user_query]).tolist()
            
            # Query ChromaDB for top 3 matching chunks
            search_results = collection.query(
                query_embeddings=query_vector,
                n_results=3
            )
            
            st.subheader("Top 3 Semantically Retrieved Chunks from ChromaDB:")
            
            retrieved_docs = search_results["documents"][0]
            retrieved_meta = search_results["metadatas"][0]
            retrieved_dist = search_results["distances"][0]
            
            for rank, (doc_text, meta, dist) in enumerate(zip(retrieved_docs, retrieved_meta, retrieved_dist), start=1):
                # Convert Cosine Distance to Cosine Similarity Score
                similarity_score = 1.0 - dist
                
                with st.expander(f"Rank {rank} | Cosine Similarity: {similarity_score:.4f} (Distance: {dist:.4f})", expanded=True):
                    st.write(doc_text)
                    st.caption(f"📍 **Source File:** `{meta['filename']}` | **Chunk Index:** `{meta['chunk_index']}`")

    st.markdown("---")
    st.subheader("🧠 Concept: Semantic Search vs. Keyword Search")
    st.write("""
    * **Keyword Search (e.g., BM25, TF-IDF):** Relies on exact string matching. If a user queries *"vehicle"*, keyword search misses documents containing only *"car"*.
    * **Semantic Search (ChromaDB + SentenceTransformers):** Maps text into a high-dimensional vector space (`all-MiniLM-L6-v2`). Matches query intent based on **underlying conceptual meaning** and vector distance (Cosine Similarity), regardless of exact keyword overlap.
    """)