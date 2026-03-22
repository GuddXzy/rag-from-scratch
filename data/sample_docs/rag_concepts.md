# Retrieval-Augmented Generation (RAG)

## What is RAG?

RAG is an AI framework for retrieving facts from an external knowledge base to ground large language models on the most accurate, up-to-date information. It combines the power of retrieval systems with generative AI models.

## RAG Architecture

A typical RAG pipeline has two main components:

1. **Indexing Pipeline**: Documents → Split into chunks → Embed chunks → Store in vector database
2. **Retrieval Pipeline**: User query → Embed query → Search vector DB → Get relevant chunks → Pass to LLM → Generate answer

## Document Loaders

LangChain provides many document loaders for different file formats:

- `PyPDFLoader` - Load PDF files
- `TextLoader` - Load plain text files  
- `WebBaseLoader` - Load web pages
- `CSVLoader` - Load CSV files
- `UnstructuredMarkdownLoader` - Load Markdown files

Example usage:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()
```

## Text Splitters

After loading documents, you typically split them into smaller chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_documents(pages)
```

The `RecursiveCharacterTextSplitter` is recommended for generic text. It tries to split on natural boundaries like paragraphs and sentences before falling back to character count.

## Vector Stores

LangChain supports many vector stores:

- **Chroma** - Lightweight, in-process, great for prototyping
- **Pinecone** - Managed cloud service, scales well
- **FAISS** - Facebook's similarity search library, very fast
- **Qdrant** - Open-source, feature-rich

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# Search
results = vectorstore.similarity_search("What is RAG?", k=3)
```

## Retrieval Strategies

- **Similarity Search**: Basic cosine similarity between query and document embeddings
- **MMR (Maximal Marginal Relevance)**: Balances relevance with diversity
- **Multi-Query Retrieval**: Generate multiple query variants to improve recall
- **Contextual Compression**: Compress retrieved documents to only relevant parts
