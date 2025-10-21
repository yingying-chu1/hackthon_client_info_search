"""
SQLite database configuration and models for structured and unstructured client information.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import os
import json
from typing import Dict, Any, Optional, List

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./client_info.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database models
class Client(Base):
    """Client model for storing structured and unstructured client information."""
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Structured fields (commonly used)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(50), index=True)
    company = Column(String(255), index=True)
    job_title = Column(String(255))
    industry = Column(String(100))
    location = Column(String(255))
    website = Column(String(500))
    
    # Status and classification
    status = Column(String(50), default="active", index=True)  # active, inactive, prospect, lead
    priority = Column(String(20), default="medium")  # low, medium, high, urgent
    source = Column(String(100))  # referral, website, cold_call, etc.
    
    # Financial information
    budget_range = Column(String(50))  # $0-10k, $10k-50k, etc.
    annual_revenue = Column(Float)
    
    # Communication preferences
    preferred_contact_method = Column(String(50))  # email, phone, sms
    timezone = Column(String(50))
    language = Column(String(10), default="en")
    
    # Unstructured data storage
    notes = Column(Text)  # Free-form notes
    raw_data = Column(JSON, default=dict)  # Store any unstructured data
    custom_fields = Column(JSON, default=dict)  # Custom fields for specific use cases
    
    # Metadata and tracking
    tags = Column(JSON, default=list)  # Flexible tagging system
    extra_metadata = Column(JSON, default=dict)  # Additional metadata
    last_contact_date = Column(DateTime)
    next_follow_up = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship("ClientInteraction", back_populates="client", cascade="all, delete-orphan")
    documents = relationship("ClientDocument", back_populates="client", cascade="all, delete-orphan")

class ClientInteraction(Base):
    """Model for tracking client interactions and communications."""
    __tablename__ = "client_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    
    # Interaction details
    interaction_type = Column(String(50), nullable=False, index=True)  # call, email, meeting, note
    subject = Column(String(255))
    content = Column(Text)
    outcome = Column(String(100))  # successful, no_answer, callback_requested, etc.
    
    # Participants and context
    participants = Column(JSON, default=list)  # List of people involved
    location = Column(String(255))  # Meeting location or call type
    duration_minutes = Column(Integer)
    
    # Follow-up information
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(DateTime)
    follow_up_notes = Column(Text)
    
    # Unstructured data
    raw_data = Column(JSON, default=dict)  # Store any additional unstructured data
    attachments = Column(JSON, default=list)  # File attachments or references
    
    # Metadata
    tags = Column(JSON, default=list)
    extra_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="interactions")

class ClientDocument(Base):
    """Model for storing documents related to clients."""
    __tablename__ = "client_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    
    # Document information
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    document_type = Column(String(50), index=True)  # contract, proposal, invoice, notes, etc.
    file_path = Column(String(500))  # Path to actual file if stored separately
    file_size = Column(Integer)
    mime_type = Column(String(100))
    
    # Content analysis
    summary = Column(Text)  # AI-generated summary
    keywords = Column(JSON, default=list)  # Extracted keywords
    entities = Column(JSON, default=list)  # Named entities (people, companies, etc.)
    sentiment = Column(String(20))  # positive, negative, neutral
    
    # Classification and organization
    category = Column(String(100), index=True)
    subcategory = Column(String(100))
    priority = Column(String(20), default="medium")
    confidential = Column(Boolean, default=False)
    
    # Unstructured data
    raw_data = Column(JSON, default=dict)  # Store any additional unstructured data
    custom_fields = Column(JSON, default=dict)  # Custom fields for specific document types
    
    # Metadata
    tags = Column(JSON, default=list)
    extra_metadata = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    client = relationship("Client", back_populates="documents")

class Document(Base):
    """Legacy document model for backward compatibility."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    extra_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ClientInfoSearch(Base):
    """Model for storing search queries and results for analytics."""
    __tablename__ = "client_info_searches"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    search_type = Column(String(50), index=True)  # client_search, document_search, interaction_search
    filters = Column(JSON, default=dict)
    results_count = Column(Integer)
    execution_time_ms = Column(Integer)
    
    # User context
    user_id = Column(String(100))  # If you have user authentication
    session_id = Column(String(100))
    
    # Results metadata
    results_summary = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

# Database dependency
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)

# Utility functions for handling structured and unstructured data
class ClientDataHandler:
    """Utility class for handling structured and unstructured client data."""
    
    @staticmethod
    def create_client_from_dict(data: Dict[str, Any]) -> Client:
        """Create a Client instance from a dictionary, handling both structured and unstructured data."""
        # Extract structured fields
        structured_fields = {
            'name', 'email', 'phone', 'company', 'job_title', 'industry', 
            'location', 'website', 'status', 'priority', 'source', 'budget_range',
            'annual_revenue', 'preferred_contact_method', 'timezone', 'language',
            'notes', 'last_contact_date', 'next_follow_up'
        }
        
        client_data = {}
        unstructured_data = {}
        custom_fields = {}
        
        for key, value in data.items():
            if key in structured_fields:
                client_data[key] = value
            elif key.startswith('custom_'):
                custom_fields[key] = value
            else:
                unstructured_data[key] = value
        
        # Create client instance
        client = Client(**client_data)
        
        # Store unstructured data
        if unstructured_data:
            client.raw_data = unstructured_data
        if custom_fields:
            client.custom_fields = custom_fields
            
        return client
    
    @staticmethod
    def update_client_from_dict(client: Client, data: Dict[str, Any]) -> Client:
        """Update a Client instance from a dictionary."""
        structured_fields = {
            'name', 'email', 'phone', 'company', 'job_title', 'industry', 
            'location', 'website', 'status', 'priority', 'source', 'budget_range',
            'annual_revenue', 'preferred_contact_method', 'timezone', 'language',
            'notes', 'last_contact_date', 'next_follow_up'
        }
        
        unstructured_data = client.raw_data or {}
        custom_fields = client.custom_fields or {}
        
        for key, value in data.items():
            if key in structured_fields:
                setattr(client, key, value)
            elif key.startswith('custom_'):
                custom_fields[key] = value
            else:
                unstructured_data[key] = value
        
        client.raw_data = unstructured_data
        client.custom_fields = custom_fields
        
        return client
    
    @staticmethod
    def search_clients_by_unstructured_data(db: Session, search_term: str, 
                                          search_fields: List[str] = None) -> List[Client]:
        """Search clients by unstructured data in raw_data and custom_fields."""
        query = db.query(Client)
        
        if search_fields is None:
            search_fields = ['raw_data', 'custom_fields', 'notes']
        
        conditions = []
        for field in search_fields:
            if field == 'raw_data':
                conditions.append(Client.raw_data.contains(search_term))
            elif field == 'custom_fields':
                conditions.append(Client.custom_fields.contains(search_term))
            elif field == 'notes':
                conditions.append(Client.notes.contains(search_term))
        
        if conditions:
            from sqlalchemy import or_
            query = query.filter(or_(*conditions))
        
        return query.all()
    
    @staticmethod
    def get_client_summary(client: Client) -> Dict[str, Any]:
        """Get a comprehensive summary of client data including structured and unstructured."""
        summary = {
            'id': client.id,
            'name': client.name,
            'email': client.email,
            'company': client.company,
            'status': client.status,
            'priority': client.priority,
            'last_contact': client.last_contact_date,
            'next_follow_up': client.next_follow_up,
            'interaction_count': len(client.interactions) if client.interactions else 0,
            'document_count': len(client.documents) if client.documents else 0,
            'tags': client.tags or [],
            'unstructured_fields': list(client.raw_data.keys()) if client.raw_data else [],
            'custom_fields': list(client.custom_fields.keys()) if client.custom_fields else []
        }
        
        return summary

class DocumentDataHandler:
    """Utility class for handling document data."""
    
    @staticmethod
    def create_document_from_dict(data: Dict[str, Any], client_id: int = None) -> ClientDocument:
        """Create a ClientDocument instance from a dictionary."""
        structured_fields = {
            'title', 'content', 'document_type', 'file_path', 'file_size',
            'mime_type', 'summary', 'keywords', 'entities', 'sentiment',
            'category', 'subcategory', 'priority', 'confidential'
        }
        
        document_data = {'client_id': client_id} if client_id else {}
        unstructured_data = {}
        custom_fields = {}
        
        for key, value in data.items():
            if key in structured_fields:
                document_data[key] = value
            elif key.startswith('custom_'):
                custom_fields[key] = value
            else:
                unstructured_data[key] = value
        
        document = ClientDocument(**document_data)
        
        if unstructured_data:
            document.raw_data = unstructured_data
        if custom_fields:
            document.custom_fields = custom_fields
            
        return document
    
    @staticmethod
    def extract_entities_from_content(content: str) -> List[str]:
        """Extract potential entities from document content (basic implementation)."""
        # This is a basic implementation - you might want to use NLP libraries
        import re
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', content)
        
        # Extract potential company names (words that start with capital letters)
        companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        entities = {
            'emails': emails,
            'phones': phones,
            'companies': companies[:10]  # Limit to first 10
        }
        
        return entities

# Database query helpers
def get_clients_by_status(db: Session, status: str) -> List[Client]:
    """Get all clients with a specific status."""
    return db.query(Client).filter(Client.status == status).all()

def get_clients_by_priority(db: Session, priority: str) -> List[Client]:
    """Get all clients with a specific priority."""
    return db.query(Client).filter(Client.priority == priority).all()

def get_clients_with_follow_up_due(db: Session) -> List[Client]:
    """Get clients with follow-up dates due."""
    from datetime import datetime
    return db.query(Client).filter(
        Client.next_follow_up <= datetime.utcnow(),
        Client.next_follow_up.isnot(None)
    ).all()

def get_client_interactions_by_type(db: Session, client_id: int, interaction_type: str) -> List[ClientInteraction]:
    """Get all interactions of a specific type for a client."""
    return db.query(ClientInteraction).filter(
        ClientInteraction.client_id == client_id,
        ClientInteraction.interaction_type == interaction_type
    ).all()

def search_documents_by_content(db: Session, search_term: str) -> List[ClientDocument]:
    """Search documents by content."""
    return db.query(ClientDocument).filter(
        ClientDocument.content.contains(search_term)
    ).all()
