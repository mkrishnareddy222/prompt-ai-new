import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Core imports for Cloud Providers
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = 3               # Increased to 3 for better context matching
CHUNK_SIZE = 500        # Balanced for dynamic file uploads
CHUNK_OVERLAP = 50

if not GROQ_API_KEY:
    sys.exit("❌ GROQ_API_KEY missing. Add it to your .env file.")

def format_docs(docs):
    """Formats retrieved chunks cleanly for the LLM prompt."""
    return "\n\n".join(f"[Source: {d.metadata.get('source', 'Uploaded File')}]\n{d.page_content.strip()}" for d in docs)

# ═══════════════════════════════════════════════════════════════
# MAIN ENGINE: Dynamic Document Processing & RAG Pipeline
# ═══════════════════════════════════════════════════════════════
def process_ui_document_and_ask(uploaded_docs: list[Document], user_question: str, provider: str = "gemini"):
    """
    Call this function inside your UI route.
    
    :param uploaded_docs: A list of LangChain Document objects sent from the UI upload component.
    :param user_question: The string question asked by the user in the chat input.
    :param provider: The string selected from the UI dropdown ('gemini' or 'cohere').
    """
    if not uploaded_docs:
        return "Please upload a document first.", []

    # 1. Chunk the dynamically uploaded text
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(uploaded_docs)

    # 2. Safely initialize the user's selected cloud embedding provider
    if provider.lower() == "cohere":
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY missing in your server .env configuration.")
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        
    else:  # Default to Gemini
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY missing in your server .env configuration.")
        
        # Explicit setup to ensure the SDK routes correctly without 404 errors
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
    
    # 3. Create a clean, isolated in-memory vector database for this specific query transaction
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)  
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    # 4. Assemble the LCEL Execution Chain
    llm = ChatGroq(api_key=GROQ_API_KEY, model=MODEL, temperature=0)
    
    RAG_PROMPT = ChatPromptTemplate.from_template(
        """You are OPSAI. Answer the user's question using ONLY the provided context from their uploaded document.
If the answer cannot be found within the context, respond with: "Not found in the uploaded document."
Always cite the specific page or source if available in the context metadata.

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
    
    # 5. Execute the chain and extract sources
    result = rag_chain.invoke(user_question)
    sources = sorted({d.metadata.get("source", "Uploaded File") for d in result["sources"]})
    
    return result["answer"], sources
