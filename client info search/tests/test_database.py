"""
Tests for database models and operations.
"""

import pytest
from sqlalchemy.orm import Session
from database import Document, Client, init_db
from datetime import datetime

def test_document_model(db_session):
    """Test Document model creation and attributes."""
    document = Document(
        title="Test Document",
        content="This is test content",
        metadata={"category": "test"}
    )
    
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    
    assert document.id is not None
    assert document.title == "Test Document"
    assert document.content == "This is test content"
    assert document.metadata == {"category": "test"}
    assert isinstance(document.created_at, datetime)
    assert isinstance(document.updated_at, datetime)

def test_client_model(db_session):
    """Test Client model creation and attributes."""
    client = Client(
        name="John Doe",
        email="john@example.com",
        phone="123-456-7890",
        company="Test Company",
        notes="Test client"
    )
    
    db_session.add(client)
    db_session.commit()
    db_session.refresh(client)
    
    assert client.id is not None
    assert client.name == "John Doe"
    assert client.email == "john@example.com"
    assert client.phone == "123-456-7890"
    assert client.company == "Test Company"
    assert client.notes == "Test client"
    assert isinstance(client.created_at, datetime)
    assert isinstance(client.updated_at, datetime)

def test_client_email_unique_constraint(db_session):
    """Test that client email must be unique."""
    client1 = Client(
        name="John Doe",
        email="john@example.com"
    )
    
    client2 = Client(
        name="Jane Doe",
        email="john@example.com"  # Same email
    )
    
    db_session.add(client1)
    db_session.commit()
    
    db_session.add(client2)
    
    with pytest.raises(Exception):  # Should raise integrity error
        db_session.commit()

def test_init_db():
    """Test database initialization."""
    # This should not raise an exception
    init_db()
