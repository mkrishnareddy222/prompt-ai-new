import io
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import docx2txt
import pypdf
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PERSISTENT_DB_DIR = Path(__file__).resolve().parents[2] / "storage" / "vector_sessions"
PERSISTENT_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding_provider(provider: str):
    provider_name = (provider or "gemini").lower()

    if provider_name == "cohere":
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY missing in server configurations.")
        return CohereEmbeddings(model="embed-english-v3.0")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY missing in server configurations.")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in (".txt", ".md"):
        return file_bytes.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        pdf_stream = io.BytesIO(file_bytes)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        pages = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    if suffix in (".docx", ".doc"):
        docx_stream = io.BytesIO(file_bytes)
        return docx2txt.process(docx_stream)

    return file_bytes.decode("utf-8", errors="ignore")


def parse_bytes_to_documents(file_bytes: bytes, filename: str, session_id: str) -> List[Document]:
    suffix = Path(filename).suffix.lower()
    documents: List[Document] = []

    if suffix == ".pdf":
        pdf_stream = io.BytesIO(file_bytes)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            text_content = page.extract_text()
            if not text_content or not text_content.strip():
                continue
            metadata = {
                "source": filename,
                "session_id": session_id,
                "file_type": suffix or "unknown",
                "page_number": page_number,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            documents.append(Document(page_content=text_content, metadata=metadata))
    else:
        text_content = extract_text_from_bytes(file_bytes=file_bytes, filename=filename)
        if not text_content.strip():
            raise ValueError("File content is completely empty.")

        metadata = {
            "source": filename,
            "session_id": session_id,
            "file_type": suffix or "unknown",
            "page_number": 1,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        documents.append(Document(page_content=text_content, metadata=metadata))

    if not documents:
        raise ValueError("File content is completely empty.")

    return documents


def parse_bytes_to_document(file_bytes: bytes, filename: str, session_id: str) -> Document:
    documents = parse_bytes_to_documents(file_bytes=file_bytes, filename=filename, session_id=session_id)
    return documents[0]


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["chunk_index"] = chunk_index
        metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata = metadata

    return chunks


def save_document_chunks_to_vector_store(session_id: str, chunks: List[Document], embeddings):
    session_path = PERSISTENT_DB_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    if any(session_path.iterdir()):
        vectordb = Chroma(persist_directory=str(session_path), embedding_function=embeddings)
        vectordb.add_documents(documents=chunks)
    else:
        Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(session_path))

    return session_path


def cleanup_session_storage(session_path: Path):
    try:
        if session_path.exists():
            shutil.rmtree(session_path)
    except Exception:
        raise
