"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

def test_create_document(client, db_session):
    """Test creating a document."""
    document_data = {
        "title": "Test Document",
        "content": "This is test content",
        "metadata": {"category": "test"}
    }
    
    with patch('main.rag_service') as mock_rag_service:
        mock_rag_instance = AsyncMock()
        mock_rag_instance.add_document.return_value = True
        mock_rag_service.return_value = mock_rag_instance
        
        response = client.post("/documents/", json=document_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Document"
        assert data["content"] == "This is test content"
        assert data["metadata"] == {"category": "test"}

def test_get_documents(client, db_session):
    """Test getting all documents."""
    response = client.get("/documents/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_document_by_id(client, db_session):
    """Test getting a specific document by ID."""
    # First create a document
    document_data = {
        "title": "Test Document",
        "content": "This is test content",
        "metadata": {"category": "test"}
    }
    
    with patch('main.rag_service') as mock_rag_service:
        mock_rag_instance = AsyncMock()
        mock_rag_instance.add_document.return_value = True
        mock_rag_service.return_value = mock_rag_instance
        
        create_response = client.post("/documents/", json=document_data)
        document_id = create_response.json()["id"]
        
        # Now get the document
        response = client.get(f"/documents/{document_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == document_id
        assert data["title"] == "Test Document"

def test_get_nonexistent_document(client):
    """Test getting a non-existent document."""
    response = client.get("/documents/99999")
    assert response.status_code == 404

def test_create_client(client, db_session):
    """Test creating a client."""
    client_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "company": "Test Company",
        "notes": "Test client"
    }
    
    response = client.post("/clients/", json=client_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "John Doe"
    assert data["email"] == "john@example.com"

def test_get_clients(client, db_session):
    """Test getting all clients."""
    response = client.get("/clients/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_client_by_id(client, db_session):
    """Test getting a specific client by ID."""
    # First create a client
    client_data = {
        "name": "John Doe",
        "email": "john@example.com"
    }
    
    create_response = client.post("/clients/", json=client_data)
    client_id = create_response.json()["id"]
    
    # Now get the client
    response = client.get(f"/clients/{client_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == client_id
    assert data["name"] == "John Doe"

def test_search_documents(client):
    """Test searching documents."""
    search_data = {
        "query": "test query",
        "n_results": 5
    }
    
    with patch('main.rag_service') as mock_rag_service:
        mock_rag_instance = AsyncMock()
        mock_rag_instance.search.return_value = []
        mock_rag_service.return_value = mock_rag_instance
        
        response = client.post("/search/", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["total_results"] == 0

def test_function_call(client):
    """Test OpenAI function calling."""
    function_data = {
        "function_name": "search_documents",
        "parameters": {"query": "test"}
    }
    
    with patch('main.openai_service') as mock_openai_service:
        mock_openai_instance = AsyncMock()
        mock_openai_instance.call_function.return_value = {"result": "test"}
        mock_openai_service.return_value = mock_openai_instance
        
        response = client.post("/function-call/", json=function_data)
        assert response.status_code == 200
        data = response.json()
        assert data["function_name"] == "search_documents"
        assert data["success"] is True

def test_list_available_functions(client):
    """Test listing available functions."""
    with patch('main.openai_service') as mock_openai_service:
        mock_openai_instance = AsyncMock()
        mock_openai_instance.get_available_functions.return_value = ["test_function"]
        mock_openai_service.return_value = mock_openai_instance
        
        response = client.get("/functions/")
        assert response.status_code == 200
        data = response.json()
        assert "functions" in data
        assert "test_function" in data["functions"]
