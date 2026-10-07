import os
import logging
from typing import List, Optional
from pathlib import Path
import io
import shutil

from fastapi import APIRouter, HTTPException, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel, Field

# Core LangChain & Cloud Vector tools
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parsers for multi-format text ingestion
import pypdf
import docx2txt

from app.config.settings import settings

router = APIRouter(prefix="/api", tags=["rag"])
logger = logging.getLogger(__name__)

TOP_K = 3               
CHUNK_SIZE = 500        
CHUNK_OVERLAP = 50

# Target directory path where persistent vector storage files reside
PERSISTENT_DB_DIR = Path(__file__).parent.parent / "storage" / "vector_sessions"
PERSISTENT_DB_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════
class RAGResponse(BaseModel):
    role: str = "assistant"
    content: str
    sources: List[str] = Field(default_factory=list)
    provider: str
    session_id: str

# ═══════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════
def get_embedding_provider(provider: str):
    if provider.lower() == "cohere":
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY missing in server configurations.")
        return CohereEmbeddings(model="embed-english-v3.0")
    else:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY missing in server configurations.")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )

def parse_bytes_to_document(file_bytes: bytes, filename: str) -> Document:
    suffix = Path(filename).suffix.lower()
    text_content = ""
    try:
        if suffix in (".txt", ".md"):
            text_content = file_bytes.decode("utf-8", errors="ignore")
        elif suffix == ".pdf":
            pdf_stream = io.BytesIO(file_bytes)
            pdf_reader = pypdf.PdfReader(pdf_stream)
            text_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif suffix in (".docx", ".doc"):
            docx_stream = io.BytesIO(file_bytes)
            text_content = docx2txt.process(docx_stream)
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            
        if not text_content.strip():
            raise ValueError("File content is completely empty.")
        return Document(page_content=text_content, metadata={"source": filename})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process '{filename}': {str(exc)}")

def format_docs(docs):
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Context Link')}]\n{d.page_content.strip()}" for d in docs)

def background_disk_cleanup(session_path: Path):
    """Safely executes physical deletion of vector binary data indexes in the background."""
    try:
        if session_path.exists():
            shutil.rmtree(session_path)
            logger.info("Successfully wiped persistent vector store directory: %s", session_path)
    except Exception as e:
        logger.error("Failed executing storage purge routine for path %s: %s", session_path, str(e))


# ═══════════════════════════════════════════════════════════════
# 1. ENDPOINT A: UPLOAD & APPEND (Supports mid-chat file additions)
# ═══════════════════════════════════════════════════════════════
@router.post("/rag/upload")
async def upload_or_append_documents(
    session_id: str = Form(...),
    provider: str = Form("gemini"),
    files: List[UploadFile] = File(...)
):
    """
    Accepts  files and updates the database directory.
    - If session doesn't exist: Creates a fresh persistent folder.
    - If session exists: Dynamically updates the database with new chunks mid-conversation!
    """
    logger.info("Processing file indexing event for session: %s", session_id)
    session_path = PERSISTENT_DB_DIR / session_id

    parsed_documents = []
    for file in files:
        file_bytes = await file.read()
        doc_obj = parse_bytes_to_document(file_bytes=file_bytes, filename=file.filename)
        parsed_documents.append(doc_obj)

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(parsed_documents)
        embeddings = get_embedding_provider(provider)
        
        if session_path.exists():
            # IMPLEMENTATION VALUE: Dynamically append new files to the existing database index!
            logger.info("Session database path exists. Appending new file partitions cleanly to disk storage...")
            vectordb = Chroma(persist_directory=str(session_path), embedding_function=embeddings)
            vectordb.add_documents(documents=chunks)
            message = f"Successfully appended {len(files)} new files to active session {session_id}."
        else:
            # Create a completely fresh persistent instance mapping
            logger.info("Initializing brand new session store cluster on disk location storage layout...")
            Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(session_path))
            message = f"Successfully initialized vector store and indexed {len(files)} files for session {session_id}."
            
        return {"status": "success", "message": message}
        
    except Exception as exc:
        logger.exception("Ingestion runtime routine crashed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Failed to compile document mapping onto server storage disk: {str(exc)}")


# ═══════════════════════════════════════════════════════════════
# 2. ENDPOINT B: MULTI-TURN CONTINUOUS QUERY ROUTE
# ═══════════════════════════════════════════════════════════════
@router.post("/rag/query")
async def query_session_documents(
    session_id: str = Form(...),
    message: str = Form(...),
    provider: str = Form("gemini"),
    temperature: float = Form(0.0),
    max_tokens: int = Form(1024)
):
    logger.info("Executing conversation query search inside data cluster path for session: %s", session_id)
    session_path = PERSISTENT_DB_DIR / session_id

    if not session_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="Active context database file index missing for this session. Please call /rag/upload first."
        )

    try:
        embeddings = get_embedding_provider(provider)
        vectordb = Chroma(persist_directory=str(session_path), embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=settings.GROQ_MODEL, temperature=temperature, max_tokens=max_tokens)
        
        RAG_PROMPT = ChatPromptTemplate.from_template(
            """You are OPSAI. Answer the user's question using ONLY the provided context from their uploaded documents.
If the answer cannot be found within the context data, respond with: "Not found in the uploaded documents."
Always cite the specific file source name provided in the context metadata.

Context:
{context}

Question: {question}
Answer:"""
        )

        setup_and_retrieval = RunnableParallel(
            {"sources": retriever, "question": RunnablePassthrough()}
        )

        answer_chain = (
            {
                "context": lambda x: format_docs(x["sources"]),
                "question": lambda x: x["question"],
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

        rag_chain = setup_and_retrieval.assign(answer=answer_chain)
        
        result = rag_chain.invoke(message)
        sources = sorted({d.metadata.get("source", "Uploaded File") for d in result["sources"]})
        
        return RAGResponse(
            role="assistant",
            content=result["answer"],
            sources=sources,
            provider=provider,
            session_id=session_id
        ).model_dump()

    except Exception as exc:
        logger.exception("Pipeline lookups failed for target key session matrix: %s", session_id)
        raise HTTPException(status_code=502, detail="RAG system operation failed during index data extraction loops.")


# ═══════════════════════════════════════════════════════════════
# 3. ENDPOINT C: SECURE CLEANUP (Wipes server index files instantly)
# ═══════════════════════════════════════════════════════════════
@router.post("/rag/cleanup")
async def purge_session_storage(
    session_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Triggers immediate vector storage file space cleanups.
    - Frontend fires this whenever the user explicitly clicks "End Session",
      resets their dashboard chat window, or closes out their browser context tab.
    """
    logger.info("Received manual clean request invocation instruction for session: %s", session_id)
    session_path = PERSISTENT_DB_DIR / session_id

    if not session_path.exists():
        raise HTTPException(
            status_code=404, 
            detail="No active session storage found for this session ID. Nothing to clean."
        )