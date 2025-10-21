"""
Tests for RAG service functionality.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from rag_service import RAGService
from schemas import SearchResult

@pytest.mark.asyncio
async def test_rag_service_initialization(temp_chroma_dir):
    """Test RAG service initialization."""
    with patch('chromadb.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection
        
        rag_service = RAGService()
        await rag_service.initialize()
        
        assert rag_service.client is not None
        assert rag_service.collection is not None

@pytest.mark.asyncio
async def test_add_document(mock_rag_service):
    """Test adding document to RAG service."""
    result = await mock_rag_service.add_document(
        document_id="test_id",
        content="Test content",
        metadata={"category": "test"}
    )
    
    assert result is True
    mock_rag_service.add_document.assert_called_once_with(
        document_id="test_id",
        content="Test content",
        metadata={"category": "test"}
    )

@pytest.mark.asyncio
async def test_search_documents(mock_rag_service):
    """Test searching documents."""
    # Mock search results
    mock_results = [
        SearchResult(
            document_id="1",
            content="Test content 1",
            metadata={"category": "test"},
            distance=0.1
        ),
        SearchResult(
            document_id="2",
            content="Test content 2",
            metadata={"category": "test"},
            distance=0.2
        )
    ]
    
    mock_rag_service.search.return_value = mock_results
    
    results = await mock_rag_service.search(
        query="test query",
        n_results=2
    )
    
    assert len(results) == 2
    assert results[0].document_id == "1"
    assert results[1].document_id == "2"

@pytest.mark.asyncio
async def test_update_document(mock_rag_service):
    """Test updating document in RAG service."""
    result = await mock_rag_service.update_document(
        document_id="test_id",
        content="Updated content",
        metadata={"category": "updated"}
    )
    
    assert result is True
    mock_rag_service.update_document.assert_called_once_with(
        document_id="test_id",
        content="Updated content",
        metadata={"category": "updated"}
    )

@pytest.mark.asyncio
async def test_delete_document(mock_rag_service):
    """Test deleting document from RAG service."""
    result = await mock_rag_service.delete_document("test_id")
    
    assert result is True
    mock_rag_service.delete_document.assert_called_once_with("test_id")

@pytest.mark.asyncio
async def test_get_document_count(mock_rag_service):
    """Test getting document count."""
    mock_rag_service.get_document_count.return_value = 5
    
    count = await mock_rag_service.get_document_count()
    
    assert count == 5
    mock_rag_service.get_document_count.assert_called_once()

@pytest.mark.asyncio
async def test_cleanup(mock_rag_service):
    """Test RAG service cleanup."""
    await mock_rag_service.cleanup()
    
    mock_rag_service.cleanup.assert_called_once()
