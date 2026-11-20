# Local CLI Document Q&A (RAG Pipeline)

A privacy-focused, 100% local Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Ollama**, and **ChromaDB**. It ingests text documents, generates vector embeddings locally, and queries an open-weights LLM without relying on external cloud APIs.

## Architecture & Tech Stack
- **Framework:** LangChain
- **Embedding Model:** `nomic-embed-text` (via Ollama)
- **Vector Database:** ChromaDB
- **LLM:** `llama3.2` (via Ollama)
- **Language:** Python 3.12

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone <YOUR-GITHUB-REPO-URL>
   cd ai-rag-cli