"""
Document Q&A helpers.

Flow:
  1. Ingest by R2 path (used as doc_id). If already in Pinecone, skip download/embed.
  2. Otherwise download from R2 → extract → chunk → embed → upsert.
  3. Answer questions over that document with RAG (retrieve by doc_id).
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Any

import boto3
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

load_dotenv()

# --- clients (env-based; no hardcoded secrets) ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET", "")

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "pinecone-datacamp")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_NAMESPACE = os.getenv("DOC_NAMESPACE", "documents")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_LIMIT = 100

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

r2 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


# --- text extraction ---

def download_from_r2(key: str, bucket: str | None = None) -> bytes:
    """Download an object from Cloudflare R2 and return its bytes."""
    bucket = bucket or R2_BUCKET
    if not bucket:
        raise ValueError("R2 bucket is required (pass bucket= or set R2_BUCKET)")

    response = r2.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from PDF or text-like file bytes."""
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        return "\n".join(parts).strip()

    if suffix in {".txt", ".md", ".csv", ".json", ".html", ".htm"}:
        return data.decode("utf-8", errors="replace").strip()

    # Fallback: try UTF-8 text
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Unsupported file type for text extraction: {suffix or '(none)'}"
        ) from exc


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def extract_document(
    r2_key: str,
    *,
    bucket: str | None = None,
    filename: str | None = None,
) -> str:
    """
    Download a document from R2 and return its extracted text.

    `filename` is used only to pick an extractor (defaults to the R2 key basename).
    """
    data = download_from_r2(r2_key, bucket=bucket)
    name = filename or Path(r2_key).name
    return extract_text_from_bytes(data, name)


# --- indexing into Pinecone ---

def _embed_texts(texts: list[str], emb_model: str = EMBED_MODEL) -> list[list[float]]:
    response = openai_client.embeddings.create(input=texts, model=emb_model)
    return [item.embedding for item in response.data]


def is_indexed(doc_id: str, *, namespace: str = DEFAULT_NAMESPACE) -> bool:
    """True if Pinecone already has at least one chunk for this doc_id (R2 path)."""
    # Vector ids are "{doc_id}-{chunk_index}"; chunk 0 always exists after a successful ingest.
    result = index.fetch(ids=[f"{doc_id}-0"], namespace=namespace)
    return bool(result.vectors)


def index_document(
    text: str,
    *,
    doc_id: str,
    title: str | None = None,
    source: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    emb_model: str = EMBED_MODEL,
) -> dict[str, Any]:
    """
    Chunk + embed document text and upsert vectors into Pinecone.

    Returns a small summary with doc_id, chunk count, and namespace.
    """
    title = title or doc_id
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text extracted from document; nothing to index")

    upserted = 0
    for i in range(0, len(chunks), BATCH_LIMIT):
        batch = chunks[i : i + BATCH_LIMIT]
        embeds = _embed_texts(batch, emb_model=emb_model)
        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeds)):
            chunk_index = i + j
            vectors.append(
                {
                    "id": f"{doc_id}-{chunk_index}",
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "text": chunk,
                        "title": title,
                        "source": source or "",
                    },
                }
            )
        index.upsert(vectors=vectors, namespace=namespace)
        upserted += len(vectors)

    return {
        "doc_id": doc_id,
        "title": title,
        "chunks": upserted,
        "namespace": namespace,
        "skipped": False,
    }


def ingest_document_from_r2(
    r2_key: str,
    *,
    bucket: str | None = None,
    doc_id: str | None = None,
    title: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    emb_model: str = EMBED_MODEL,
    force: bool = False,
) -> dict[str, Any]:
    """
    Ingest by R2 path. doc_id defaults to the R2 key.

    If that doc_id is already in Pinecone, skip download/extract/embed unless force=True.
    """
    doc_id = doc_id or r2_key

    if not force and is_indexed(doc_id, namespace=namespace):
        return {
            "doc_id": doc_id,
            "title": title or Path(r2_key).name,
            "chunks": 0,
            "namespace": namespace,
            "skipped": True,
        }

    text = extract_document(r2_key, bucket=bucket)
    return index_document(
        text,
        doc_id=doc_id,
        title=title or Path(r2_key).name,
        source=r2_key,
        namespace=namespace,
        emb_model=emb_model,
    )


# --- retrieval + Q&A ---

def retrieve(
    query: str,
    *,
    top_k: int = 3,
    namespace: str = DEFAULT_NAMESPACE,
    emb_model: str = EMBED_MODEL,
    doc_id: str | None = None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Embed the query and fetch the most similar chunks from Pinecone.

    Optionally filter to a single document with `doc_id`.
    Returns (chunk_texts, sources) where each source is (title, source).
    """
    query_emb = _embed_texts([query], emb_model=emb_model)[0]

    query_kwargs: dict[str, Any] = {
        "vector": query_emb,
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": True,
    }
    if doc_id:
        query_kwargs["filter"] = {"doc_id": {"$eq": doc_id}}

    docs = index.query(**query_kwargs)

    retrieved_docs: list[str] = []
    sources: list[tuple[str, str]] = []
    for match in docs.get("matches", []):
        metadata = match.get("metadata") or {}
        retrieved_docs.append(metadata.get("text", ""))
        sources.append((metadata.get("title", ""), metadata.get("source", "")))

    return retrieved_docs, sources


def prompt_with_context_builder(query: str, docs: list[str]) -> str:
    delim = "\n\n---\n\n"
    prompt_start = "Answer the question based on the context below.\n\nContext:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    return prompt_start + delim.join(docs) + prompt_end


def question_answering(
    prompt: str,
    sources: list[tuple[str, str]],
    *,
    chat_model: str = CHAT_MODEL,
) -> str:
    """Generate an answer from a context-filled prompt and append sources."""
    sys_prompt = (
        "You are a helpful assistant that answers questions using only the "
        "provided context. If the context does not contain the answer, say so."
    )

    res = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer = res.choices[0].message.content.strip()
    if sources:
        answer += "\n\nSources:"
        seen: set[tuple[str, str]] = set()
        for title, source in sources:
            key = (title, source)
            if key in seen:
                continue
            seen.add(key)
            label = title or source or "document"
            answer += f"\n{label}" + (f": {source}" if source else "")
    return answer


def answer_question(
    query: str,
    *,
    top_k: int = 3,
    namespace: str = DEFAULT_NAMESPACE,
    emb_model: str = EMBED_MODEL,
    chat_model: str = CHAT_MODEL,
    doc_id: str | None = None,
) -> str:
    """
    End-to-end Q&A: retrieve relevant chunks, build a prompt, answer with the LLM.
    """
    documents, sources = retrieve(
        query,
        top_k=top_k,
        namespace=namespace,
        emb_model=emb_model,
        doc_id=doc_id,
    )
    if not documents:
        return "No relevant document context found for that question."

    prompt = prompt_with_context_builder(query, documents)
    return question_answering(prompt, sources, chat_model=chat_model)


# --- example usage ---
result = ingest_document_from_r2("notes/f40a107b-b8ac-49dd-b591-beb57c6d2f52/1100e346-7f84-48db-92ef-42b835cb5b3b/examples/10452006_ADAMS_MOTIVATION_LETTER.pdf")
# second call → skipped=True, no R2 download
print(answer_question("What school is being applied to?", doc_id=result["doc_id"]))
