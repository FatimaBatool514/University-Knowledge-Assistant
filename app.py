import json
import os
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "faiss_index" / "index.faiss"
CHUNKS_PATH = BASE_DIR / "chunks.json"
METADATA_PATH = BASE_DIR / "metadata.json"
CONFIG_PATH = BASE_DIR / "config.json"


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "embedding_dimension": 384,
        "faiss_index_type": "IndexFlatIP",
        "similarity_metric": "cosine_similarity",
        "normalized_embeddings": True,
    }


config = load_config()

EMBEDDING_MODEL_NAME = config.get(
    "embedding_model",
    DEFAULT_EMBEDDING_MODEL,
)

EMBEDDING_DIMENSION = int(
    config.get(
        "embedding_dimension",
        384,
    )
)


# ============================================================
# VALIDATE DATABASE FILES
# ============================================================

def validate_database_files():

    required_files = [
        INDEX_PATH,
        CHUNKS_PATH,
        METADATA_PATH,
    ]

    missing_files = [
        str(path.relative_to(BASE_DIR))
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "Required RAG database files are missing:\n\n"
            + "\n".join(f"- {file}" for file in missing_files)
        )
        st.stop()


validate_database_files()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    return index


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_data
def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)

    return chunks


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


# ============================================================
# INITIALIZE DATABASE
# ============================================================

index = load_faiss_index()
chunks = load_chunks()
metadata = load_metadata()
embedding_model = load_embedding_model()


# ============================================================
# DATABASE VALIDATION
# ============================================================

if index.d != EMBEDDING_DIMENSION:

    st.error(
        f"Embedding dimension mismatch.\n\n"
        f"FAISS dimension: {index.d}\n"
        f"Configured dimension: {EMBEDDING_DIMENSION}"
    )

    st.stop()


if index.ntotal != len(chunks):

    st.error(
        "FAISS/chunks mismatch.\n\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Text chunks: {len(chunks)}"
    )

    st.stop()


if index.ntotal != len(metadata):

    st.error(
        "FAISS/metadata mismatch.\n\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Metadata records: {len(metadata)}"
    )

    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_api_key():

    # First priority: environment variable
    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        return api_key

    # Streamlit Cloud secrets fallback
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    return api_key


GROQ_API_KEY = get_groq_api_key()

if not GROQ_API_KEY:

    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add it to your environment variables or Streamlit secrets."
    )

    groq_client = None

else:

    groq_client = Groq(
        api_key=GROQ_API_KEY,
    )


# ============================================================
# QUERY EMBEDDING
# ============================================================

def create_query_embedding(query: str):

    query = query.strip()

    if not query:
        return None

    # SentenceTransformers 6.x supports query encoding.
    # We normalize because the FAISS database was built
    # using normalized embeddings and IndexFlatIP.

    if hasattr(
        embedding_model,
        "encode_query",
    ):

        embedding = embedding_model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    else:

        embedding = embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim == 1:
        embedding = embedding.reshape(
            1,
            -1,
        )

    return embedding


# ============================================================
# FAISS SEARCH
# ============================================================

def search_documents(
    query: str,
    top_k: int = DEFAULT_TOP_K,
):

    query_embedding = create_query_embedding(
        query
    )

    if query_embedding is None:
        return []

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx < 0:
            continue

        idx = int(idx)

        # Retrieve actual text
        chunk_record = chunks[idx]

        # Retrieve traceability metadata
        metadata_record = metadata.get(
            str(idx),
            {},
        )

        results.append(
            {
                "faiss_id": idx,
                "score": float(score),
                "text": chunk_record.get(
                    "text",
                    "",
                ),
                "metadata": metadata_record,
            }
        )

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):

        md = result["metadata"]

        source = md.get(
            "source",
            "Unknown source",
        )

        page = md.get(
            "page_number",
            md.get(
                "page",
                "Unknown",
            ),
        )

        chunk_number = md.get(
            "chunk_number",
            "Unknown",
        )

        text = result["text"]

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}
Chunk: {chunk_number}

Content:
{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(
    question: str,
    results,
):

    if not results:
        return (
            "I could not find relevant information "
            "in the university knowledge base."
        )

    if groq_client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please configure the API key before asking questions."
        )

    context = build_context(
        results
    )

    system_prompt = """
You are the University Knowledge Assistant.

Your job is to answer questions using ONLY the
provided university knowledge-base context.

Rules:

1. Do not invent university policies.
2. Do not use information that is not present in the context.
3. If the answer is not available in the context,
   clearly say that the information was not found.
4. Give concise but complete answers.
5. When possible, mention the relevant document and page.
6. If multiple documents contain relevant information,
   synthesize them carefully.
7. Do not claim that a policy exists unless it appears
   in the provided context.
8. Preserve important conditions, requirements,
   deadlines, percentages, fees, and exceptions.
"""

    user_prompt = f"""
University Knowledge Base Context:

{context}

User Question:

{question}

Answer the question using only the context above.
"""

    try:

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            model=DEFAULT_GROQ_MODEL,
        )

        answer = (
            chat_completion
            .choices[0]
            .message
            .content
        )

        return answer

    except Exception as e:

        return (
            "An error occurred while contacting Groq: "
            f"{str(e)}"
        )


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(results):

    if not results:
        return

    st.subheader("📚 Sources")

    displayed_sources = set()

    for i, result in enumerate(
        results,
        start=1,
    ):

        md = result["metadata"]

        source = md.get(
            "source",
            "Unknown source",
        )

        page = md.get(
            "page_number",
            md.get(
                "page",
                "Unknown",
            ),
        )

        chunk_number = md.get(
            "chunk_number",
            "Unknown",
        )

        score = result["score"]

        source_key = (
            source,
            page,
            chunk_number,
        )

        if source_key in displayed_sources:
            continue

        displayed_sources.add(
            source_key
        )

        with st.expander(
            f"{source} — Page {page}"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption(
                    f"FAISS ID: {result['faiss_id']}"
                )

            with col2:
                st.caption(
                    f"Chunk: {chunk_number}"
                )

            with col3:
                st.caption(
                    f"Similarity: {score:.4f}"
                )

            st.write(
                result["text"]
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 University Assistant")

    st.markdown(
        """
Ask questions about:

- Student Handbook
- Scholarship Policy
- Fee Policy
- Examination Rules
- Admission Guidelines
- Academic Calendar
        """
    )

    st.divider()

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
    )

    st.divider()

    st.caption(
        "RAG Architecture"
    )

    st.caption(
        "FAISS → Metadata → Groq"
    )

    st.caption(
        f"Embedding: {EMBEDDING_MODEL_NAME}"
    )

    st.caption(
        f"LLM: {DEFAULT_GROQ_MODEL}"
    )


# ============================================================
# MAIN UI
# ============================================================

st.title(
    "🎓 University Student & Academic Knowledge Assistant"
)

st.markdown(
    """
Ask questions about university academic policies,
fees, scholarships, examinations, admissions,
and the academic calendar.
"""
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about university policies..."
)


if question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # Retrieve documents
    with st.spinner(
        "Searching the university knowledge base..."
    ):

        results = search_documents(
            question,
            top_k=top_k,
        )

    # Generate response
    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                results,
            )

        st.markdown(
            answer
        )

        display_sources(
            results
        )

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
