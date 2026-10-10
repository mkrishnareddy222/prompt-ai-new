import logging
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.services.ingestion import (PERSISTENT_DB_DIR, chunk_documents,
                                    cleanup_session_storage,
                                    get_embedding_provider,
                                    parse_bytes_to_documents,
                                    save_document_chunks_to_vector_store)
from rag.services.retrieval import answer_question

router = APIRouter(prefix="/api", tags=["rag"])
logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    role: str = "assistant"
    content: str
    sources: List[str] = Field(default_factory=list)
    metadata: List[Dict[str, Any]] = Field(default_factory=list)
    provider: str
    session_id: str


@router.post("/rag/upload")
async def upload_or_append_documents(
    session_id: str = Form(...),
    provider: str = Form("gemini"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    files: List[UploadFile] = File(...),
):
    logger.info(
        "Processing file indexing event for session %s (Chunk Size: %d, Overlap: %d)",
        session_id,
        chunk_size,
        chunk_overlap,
    )

    parsed_documents = []
    for file in files:
        file_bytes = await file.read()
        parsed_documents.extend(parse_bytes_to_documents(file_bytes=file_bytes, filename=file.filename, session_id=session_id))

    try:
        chunks = chunk_documents(parsed_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = get_embedding_provider(provider)
        save_document_chunks_to_vector_store(session_id=session_id, chunks=chunks, embeddings=embeddings)

        return {
            "status": "success",
            "message": f"Successfully indexed {len(files)} file(s) for session {session_id}.",
            "session_id": session_id,
            "chunk_count": len(chunks),
        }
    except Exception as exc:
        logger.exception("Ingestion runtime routine crashed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Failed to compile document mapping onto server storage disk: {str(exc)}")


@router.post("/rag/query")
async def query_session_documents(
    session_id: str = Form(...),
    message: str = Form(...),
    provider: str = Form("gemini"),
    temperature: float = Form(0.0),
    max_tokens: int = Form(1024),
):
    logger.info("Executing conversation query search inside data cluster path for session: %s", session_id)

    try:
        result = answer_question(
            session_id=session_id,
            message=message,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return RAGResponse(
            role="assistant",
            content=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"],
            provider=provider,
            session_id=session_id,
        ).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline lookups failed for target key session matrix: %s", session_id)
        raise HTTPException(status_code=502, detail=f"RAG system operation failed during index data extraction loops: {str(exc)}")


@router.post("/rag/cleanup")
async def purge_session_storage(
    session_id: str = Form(...),
    background_tasks: BackgroundTasks = None,
):
    logger.info("Received manual clean request invocation instruction for session: %s", session_id)
    session_path = PERSISTENT_DB_DIR / session_id

    if not session_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No active session storage found for this session ID. Nothing to clean.",
        )

    if background_tasks is None:
        background_tasks = BackgroundTasks()

    background_tasks.add_task(cleanup_session_storage, session_path)
    return {"status": "success", "message": f"Session storage container cleanup scheduled for {session_id}."}