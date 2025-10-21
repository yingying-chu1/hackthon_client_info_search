"""
Test configuration and utilities.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from main import app
from database import Base, get_db
from rag_service import RAGService
from openai_service import OpenAIService

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def db_session():
    """Create database session for testing."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def mock_rag_service():
    """Mock RAG service for testing."""
    mock_service = AsyncMock(spec=RAGService)
    mock_service.initialize = AsyncMock(return_value=None)
    mock_service.add_document = AsyncMock(return_value=True)
    mock_service.search = AsyncMock(return_value=[])
    mock_service.update_document = AsyncMock(return_value=True)
    mock_service.delete_document = AsyncMock(return_value=True)
    mock_service.get_document_count = AsyncMock(return_value=0)
    mock_service.cleanup = AsyncMock(return_value=None)
    return mock_service

@pytest.fixture
def mock_openai_service():
    """Mock OpenAI service for testing."""
    mock_service = MagicMock(spec=OpenAIService)
    mock_service.call_function = AsyncMock(return_value={"result": "test"})
    mock_service.chat_with_functions = AsyncMock(return_value={"response": "test"})
    mock_service.get_available_functions = MagicMock(return_value=["test_function"])
    mock_service.get_function_schema = MagicMock(return_value={"name": "test_function"})
    mock_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_service.moderate_text = AsyncMock(return_value={"flagged": False})
    return mock_service

@pytest.fixture
def temp_chroma_dir():
    """Create temporary directory for Chroma database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
