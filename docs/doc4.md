# Semantic Search vs. Keyword Search

Searching document repositories effectively requires understanding the fundamental differences between lexical token matching and conceptual vector retrieval.

## Keyword-Based Search
Traditional search engines rely on lexical matching techniques such as BM25 or TF-IDF. 
- **Mechanism:** Matches exact character strings and token frequencies between the search query and the target index.
- **Limitations:** Fails when a user's query uses vocabulary or synonyms different from the text. For instance, searching for *"automobile repairs"* will miss passages containing only *"car maintenance"*.

## Semantic Search
Semantic search utilizes neural network models to match queries based on **context and intent**.
- **Mechanism:** Converts both the search query and target text into embedding vectors and computes distance metrics in vector space.
- **Advantages:** Captures conceptual meanings, multi-word phrasing, and synonyms, allowing relevant chunks to be retrieved even when zero keywords overlap.