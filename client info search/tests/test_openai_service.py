"""
Tests for OpenAI service functionality.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openai_service import OpenAIService

@pytest.mark.asyncio
async def test_openai_service_initialization():
    """Test OpenAI service initialization."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
        service = OpenAIService()
        assert service.api_key == 'test_key'
        assert service.model == 'gpt-3.5-turbo'

def test_openai_service_without_api_key():
    """Test OpenAI service initialization without API key."""
    with patch.dict('os.environ', {}, clear=True):
        service = OpenAIService()
        assert service.api_key is None
        assert service.client is None

def test_get_available_functions():
    """Test getting available functions."""
    service = OpenAIService()
    functions = service.get_available_functions()
    
    assert "search_documents" in functions
    assert "get_client_info" in functions
    assert "create_client" in functions
    assert "analyze_text" in functions

def test_get_function_schema():
    """Test getting function schema."""
    service = OpenAIService()
    schema = service.get_function_schema("search_documents")
    
    assert schema is not None
    assert schema["name"] == "search_documents"
    assert "parameters" in schema

def test_get_nonexistent_function_schema():
    """Test getting schema for non-existent function."""
    service = OpenAIService()
    schema = service.get_function_schema("nonexistent_function")
    
    assert schema is None

@pytest.mark.asyncio
async def test_call_function_without_client():
    """Test calling function without OpenAI client."""
    service = OpenAIService()
    service.client = None
    
    result = await service.call_function(
        function_name="search_documents",
        parameters={"query": "test"}
    )
    
    assert "error" in result
    assert result["error"] == "OpenAI client not initialized"

@pytest.mark.asyncio
async def test_call_nonexistent_function():
    """Test calling non-existent function."""
    service = OpenAIService()
    service.client = MagicMock()
    
    result = await service.call_function(
        function_name="nonexistent_function",
        parameters={"query": "test"}
    )
    
    assert "error" in result
    assert "not found" in result["error"]

@pytest.mark.asyncio
async def test_generate_embedding_without_client():
    """Test generating embedding without OpenAI client."""
    service = OpenAIService()
    service.client = None
    
    with pytest.raises(Exception) as exc_info:
        await service.generate_embedding("test text")
    
    assert "OpenAI client not initialized" in str(exc_info.value)

@pytest.mark.asyncio
async def test_moderate_text_without_client():
    """Test moderating text without OpenAI client."""
    service = OpenAIService()
    service.client = None
    
    with pytest.raises(Exception) as exc_info:
        await service.moderate_text("test text")
    
    assert "OpenAI client not initialized" in str(exc_info.value)
