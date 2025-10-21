"""
Tests for main FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Client Info Search API"
    assert data["version"] == "1.0.0"

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data

@pytest.mark.asyncio
async def test_app_lifespan():
    """Test application lifespan events."""
    from main import lifespan
    
    # Mock the services
    with patch('main.rag_service') as mock_rag, \
         patch('main.openai_service') as mock_openai, \
         patch('main.init_db') as mock_init_db:
        
        mock_rag_instance = AsyncMock()
        mock_openai_instance = AsyncMock()
        
        mock_rag.return_value = mock_rag_instance
        mock_openai.return_value = mock_openai_instance
        
        # Test lifespan context manager
        async with lifespan(None):
            pass
        
        # Verify initialization calls
        mock_init_db.assert_called_once()
        mock_rag_instance.initialize.assert_called_once()
