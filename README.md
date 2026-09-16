# University Knowledge Assistant RAG Database

Precomputed vector database for the University Student & Academic
Knowledge Assistant.

## Architecture

Google Drive PDFs
        ↓
PyMuPDF
        ↓
Page-aware Chunking
        ↓
SentenceTransformer
        ↓
FAISS
        ↓
Metadata JSON
        ↓
Streamlit + Groq

## Documents

Expected documents: 6

Actual documents: 6

## Embedding

Model:
sentence-transformers/all-MiniLM-L6-v2

Dimension:
384

## Chunking

Chunk size:
900

Chunk overlap:
150

Chunks remain within individual PDF pages so that the application
can provide accurate source/page citations.

## Database Structure

faiss_index/
    index.faiss

chunks.json
    Contains the actual text chunks.

metadata.json
    Contains metadata for every FAISS vector.

config.json
    Contains embedding and vector database configuration.

manifest.json
    Contains source document information.

source_map.json
    Lightweight FAISS ID → source mapping.

version.json
    Database version and statistics.

## Metadata

Each vector contains:

- FAISS ID
- chunk ID
- source filename
- page number
- total pages
- chunk number
- chunks on page
- source label
- document type
- collection name
- source file SHA-256
- file size
- embedding model
- embedding dimension
- chunk size
- chunk overlap
- indexing timestamp

## FAISS Mapping

FAISS ID 0
    ↓
metadata["0"]
    ↓
chunks[0]

FAISS ID 1
    ↓
metadata["1"]
    ↓
chunks[1]

This mapping must be preserved.

## Streamlit Application

The Streamlit application should LOAD this database.

Do not recreate the embeddings during normal application startup.

The application should:

1. Load index.faiss
2. Load chunks.json
3. Load metadata.json
4. Embed the user's question
5. Search FAISS
6. Retrieve the corresponding chunks
7. Retrieve metadata
8. Send retrieved context to Groq
9. Display the answer
10. Display source citations

## Example Citation

Scholarship Policy.pdf — Page 4

The metadata file allows the application to identify the exact
document and page from which retrieved information originated.

## Important

This database is intended for development/testing of a RAG
application. The underlying university documents should be treated
as the authoritative source in a production deployment.