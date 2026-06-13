"""
pipeline.py — Singleton pipeline manager.

Loads the heavy models (embeddings + CrossEncoder) once at startup,
then exposes run_match() and run_qa() for the API endpoints to call.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# LangChain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

load_dotenv()

logger = logging.getLogger("pipeline")

RESUMES_DIR = Path("data/resumes")
CHROMA_DIR = "chroma_db"

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

ATS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert AI ATS (Applicant Tracking System).

TASK:
Match each resume against the given Job Description and return STRICT JSON only.

SCORING RULES:
- 85–100 → Excellent match
- 70–84  → Strong match
- 50–69  → Medium match
- Below 50 → Weak match

OUTPUT FORMAT (return ONLY a valid JSON array, no extra text, no markdown):
[
  {{
    "rank": 1,
    "resume_file": "<filename>",
    "match_score": <integer 0-100>,
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3"],
    "evidence": "<1-2 sentence justification>"
  }},
  ...
]

Job Description:
{question}

Resumes:
{context}
""")

QA_PROMPT = ChatPromptTemplate.from_template("""
You are a Shortlisted Candidate Q&A assistant.

RULES:
- Answer ONLY using the shortlisted resumes below
- Mention the resume file name when referencing a candidate
- Compare candidates if the question asks for it
- Do NOT use any external knowledge

SHORTLISTED RESUMES:
{context}

QUESTION:
{question}
""")


# ──────────────────────────────────────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────────────────────────────────────

_embedding_model: Optional[HuggingFaceEmbeddings] = None
_reranker: Optional[CrossEncoder] = None
_llm: Optional[ChatGoogleGenerativeAI] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model…")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder reranker…")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    return _llm


def _get_vector_store() -> Optional[Chroma]:
    embeddings = _get_embedding_model()
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        docs = _load_resumes(RESUMES_DIR)
        if not docs:
            return None
        return _build_vector_store(docs)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_resumes(folder: Path):
    """Load all PDFs from folder into LangChain Documents."""
    all_docs = []
    for pdf_path in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        all_docs.extend(docs)
    return all_docs


def _build_vector_store(docs) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = _get_embedding_model()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


def _rerank_docs(query: str, docs):
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]


def _deduplicate(docs):
    seen = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen[src] = doc
    return list(seen.values())


def _build_context(docs) -> str:
    parts = []
    for doc in docs:
        parts.append(
            f"Resume File: {doc.metadata.get('source', 'unknown')}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n====================\n\n".join(parts)


def _parse_ats_response(raw: str, docs) -> list:
    """Parse JSON from LLM response; fall back to a stub on failure."""
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed, returning stub results.")
        # Build minimal stubs from available docs
        stubs = []
        for i, doc in enumerate(docs):
            stubs.append({
                "rank": i + 1,
                "resume_file": doc.metadata.get("source", f"resume_{i+1}.pdf"),
                "match_score": 50,
                "matched_skills": [],
                "missing_skills": [],
                "evidence": "Score could not be parsed from LLM response.",
            })
        return stubs


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def list_resumes() -> List[str]:
    """Return filenames of all PDFs in the resumes folder."""
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in RESUMES_DIR.glob("*.pdf"))


def run_match(job_description: str, top_k: int = 5) -> list:
    """
    Run the full RAG pipeline against the given JD.
    Returns a list of CandidateMatch-compatible dicts sorted by rank.
    """
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    docs = _load_resumes(RESUMES_DIR)
    if not docs:
        return []

    # Build fresh vector store each time (reflects newly uploaded files)
    vector_store = _build_vector_store(docs)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": min(top_k + 3, 10), "fetch_k": 20},
    )

    retrieved = retriever.invoke(job_description)
    reranked = _rerank_docs(job_description, retrieved)
    deduped = _deduplicate(reranked)[:top_k]

    context = _build_context(deduped)
    llm = _get_llm()
    messages = ATS_PROMPT.invoke({"context": context, "question": job_description})
    response = llm.invoke(messages)

    candidates = _parse_ats_response(response.content, deduped)

    # Attach raw preview
    source_to_doc = {d.metadata.get("source"): d for d in deduped}
    for c in candidates:
        doc = source_to_doc.get(c.get("resume_file"))
        if doc:
            c["raw_text_preview"] = doc.page_content[:300]

    # Ensure sorted by match_score descending and re-rank numbers
    candidates.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    return candidates


def run_qa(question: str, shortlisted_files: List[str]) -> dict:
    """
    Answer a question about specific shortlisted candidates by retrieving
    relevant chunks from ChromaDB filtered by candidate source filenames.
    """
    RESUMES_DIR.mkdir(parents=True, exist_ok=True)
    
    vector_store = _get_vector_store()
    if not vector_store:
        return {
            "answer": "No resumes have been uploaded/indexed yet.",
            "sources": [],
        }

    # Filter ChromaDB to only search within the shortlisted files
    if len(shortlisted_files) == 1:
        filter_dict = {"source": shortlisted_files[0]}
    else:
        filter_dict = {"source": {"$in": shortlisted_files}}

    # Retrieve relevant chunks from ChromaDB using metadata filter
    # k can be set to 10 to ensure we get plenty of context from the selected resumes
    retrieved_docs = vector_store.similarity_search(
        question,
        k=10,
        filter=filter_dict
    )

    if not retrieved_docs:
        return {
            "answer": "No relevant chunks found in the vector store for the selected candidates.",
            "sources": [],
        }

    # Build context from the retrieved chunks
    context = _build_context(retrieved_docs)
    llm = _get_llm()
    messages = QA_PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": list({d.metadata.get("source") for d in retrieved_docs}),
    }
