
# Retrieval-Augmented Generation (RAG)

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) is an approach that combines information retrieval with a language model. Before generating an answer, the system retrieves relevant information from an external knowledge source and provides that information as context to the model. This helps ground answers in available documents and makes it possible to update the knowledge source without retraining the language model.

## How does a RAG system retrieve relevant document chunks?

A RAG system divides documents into smaller text chunks and converts each chunk into a numerical embedding. The embeddings are stored in a vector database along with the chunk text and metadata. When a user asks a question, the system converts the question into an embedding and searches for chunks with similar vector representations. It returns the top-ranked chunks, which can then be passed to a language model as context for answer generation.

## Why are chunking and embeddings important?

Chunking divides long documents into manageable passages. Overlapping adjacent chunks can help preserve context across boundaries. Embeddings represent semantic meaning numerically, allowing a vector database to retrieve passages that are conceptually related to a question rather than relying only on exact keyword matches.
