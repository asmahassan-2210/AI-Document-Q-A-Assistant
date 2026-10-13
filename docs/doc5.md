# Retrieval-Augmented Generation (RAG) Architecture

Retrieval-Augmented Generation (RAG) is a design pattern that connects Large Language Models to external knowledge bases to eliminate hallucinations and provide up-to-date, domain-specific answers.

## RAG System Workflow

1. **Document Ingestion:** Raw documents (PDF, TXT, MD) are parsed and split into smaller overlapping text chunks (e.g., 500 characters with 50-character overlap).
2. **Embedding Generation:** Each text chunk is converted into an embedding vector using an encoder model like `all-MiniLM-L6-v2`.
3. **Vector Storage:** Embeddings, metadata, and raw text are stored in ChromaDB.
4. **Query Retrieval:** When a user asks a question, the query is embedded into a vector, and ChromaDB retrieves the top 3 semantically closest chunks.
5. **Contextual Generation:** The retrieved chunks are injected into the LLM prompt as context to generate a grounded, accurate response.