import os
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config.settings import settings
from rag.services.ingestion import PERSISTENT_DB_DIR, get_embedding_provider

TOP_K = 3


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Context Link')} | Chunk: {doc.metadata.get('chunk_index', 0)}]\n{doc.page_content.strip()}"
        for doc in docs
    )


def get_session_vector_store(session_id: str, provider: str):
    session_path = PERSISTENT_DB_DIR / session_id
    if not session_path.exists():
        raise FileNotFoundError("Active context database file index missing for this session. Please call /rag/upload first.")

    embeddings = get_embedding_provider(provider)
    return Chroma(persist_directory=str(session_path), embedding_function=embeddings)


def build_retrieval_metadata(docs) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    for doc in docs:
        meta = dict(doc.metadata or {})
        metadata.append(
            {
                "source": meta.get("source", "Uploaded File"),
                "file_type": meta.get("file_type", "unknown"),
                "page_number": meta.get("page_number", 1),
                "chunk_index": meta.get("chunk_index", 0),
                "chunk_size": meta.get("chunk_size", len(doc.page_content)),
                "session_id": meta.get("session_id", "unknown"),
            }
        )
    return metadata


def answer_question(session_id: str, message: str, provider: str, temperature: float = 0.0, max_tokens: int = 1024):
    vectordb = get_session_vector_store(session_id=session_id, provider=provider)
    relevant_docs = vectordb.similarity_search(query=message, k=TOP_K)

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=settings.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    rag_prompt = ChatPromptTemplate.from_template(
        """You are OPSAI. Answer the user's question using ONLY the provided context from their uploaded documents.
If the answer cannot be found within the context data, respond with: "Not found in the uploaded documents."
Always cite the specific file source name provided in the context metadata.

Context:
{context}

Question: {question}
Answer:"""
    )

    context_text = format_docs(relevant_docs)
    answer = (rag_prompt | llm | StrOutputParser()).invoke({"context": context_text, "question": message})
    sources = sorted({doc.metadata.get("source", "Uploaded File") for doc in relevant_docs})
    metadata = build_retrieval_metadata(relevant_docs)

    return {
        "answer": answer,
        "sources": sources,
        "metadata": metadata,
    }
