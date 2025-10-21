"""
FastAPI application with SQLite, Chroma RAG, and OpenAI function-calling tools.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
from typing import List, Dict, Any, Optional

from database import get_db, init_db
from models import Document, Client
from rag_service import RAGService
from openai_service import OpenAIService
from schemas import (
    DocumentCreate, DocumentResponse, 
    ClientCreate, ClientResponse,
    QueryRequest, QueryResponse,
    FunctionCallRequest, FunctionCallResponse
)

# Global services
rag_service: Optional[RAGService] = None
openai_service: Optional[OpenAIService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    global rag_service, openai_service
    
    # Startup
    print("Starting up...")
    init_db()
    
    # Initialize services
    rag_service = RAGService()
    await rag_service.initialize()
    
    openai_service = OpenAIService()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if rag_service:
        await rag_service.cleanup()

app = FastAPI(
    title="Client Info Search API",
    description="FastAPI app with SQLite, Chroma RAG, and OpenAI function-calling tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get RAG service
def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return rag_service

# Dependency to get OpenAI service
def get_openai_service() -> OpenAIService:
    if openai_service is None:
        raise HTTPException(status_code=503, detail="OpenAI service not initialized")
    return openai_service

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Client Info Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "services": {
        "rag": rag_service is not None,
        "openai": openai_service is not None
    }}

# Document endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db=Depends(get_db),
    rag=Depends(get_rag_service)
):
    """Create a new document and add it to the vector store."""
    try:
        # Create document in database
        db_document = Document(
            title=document.title,
            content=document.content,
            metadata=document.metadata
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Add to vector store
        await rag.add_document(
            document_id=str(db_document.id),
            content=document.content,
            metadata={**document.metadata, "title": document.title}
        )
        
        return DocumentResponse.from_orm(db_document)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/documents/", response_model=List[DocumentResponse])
async def get_documents(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    """Get all documents with pagination."""
    documents = db.query(Document).offset(skip).limit(limit).all()
    return [DocumentResponse.from_orm(doc) for doc in documents]

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db=Depends(get_db)):
    """Get a specific document by ID."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.from_orm(document)

# Client endpoints
@app.post("/clients/", response_model=ClientResponse)
async def create_client(client: ClientCreate, db=Depends(get_db)):
    """Create a new client."""
    try:
        db_client = Client(**client.dict())
        db.add(db_client)
        db.commit()
        db.refresh(db_client)
        return ClientResponse.from_orm(db_client)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/clients/", response_model=List[ClientResponse])
async def get_clients(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    """Get all clients with pagination."""
    clients = db.query(Client).offset(skip).limit(limit).all()
    return [ClientResponse.from_orm(client) for client in clients]

@app.get("/clients/{client_id}", response_model=ClientResponse)
async def get_client(client_id: int, db=Depends(get_db)):
    """Get a specific client by ID."""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return ClientResponse.from_orm(client)

# RAG search endpoints
@app.post("/search/", response_model=QueryResponse)
async def search_documents(
    query: QueryRequest,
    rag=Depends(get_rag_service)
):
    """Search documents using RAG."""
    try:
        results = await rag.search(
            query=query.query,
            n_results=query.n_results,
            filter_metadata=query.filter_metadata
        )
        return QueryResponse(
            query=query.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# OpenAI function calling endpoints
@app.post("/function-call/", response_model=FunctionCallResponse)
async def function_call(
    request: FunctionCallRequest,
    openai_service=Depends(get_openai_service)
):
    """Execute OpenAI function calling."""
    try:
        result = await openai_service.call_function(
            function_name=request.function_name,
            parameters=request.parameters,
            messages=request.messages
        )
        return FunctionCallResponse(
            function_name=request.function_name,
            result=result,
            success=True
        )
    except Exception as e:
        return FunctionCallResponse(
            function_name=request.function_name,
            result={"error": str(e)},
            success=False
        )

@app.get("/functions/")
async def list_available_functions(openai_service=Depends(get_openai_service)):
    """List all available OpenAI functions."""
    return {"functions": openai_service.get_available_functions()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
