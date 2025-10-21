"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime

# Document schemas
class DocumentBase(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Client schemas
class ClientBase(BaseModel):
    # Structured fields
    name: str
    email: EmailStr
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    
    # Status and classification
    status: Optional[str] = "active"
    priority: Optional[str] = "medium"
    source: Optional[str] = None
    
    # Financial information
    budget_range: Optional[str] = None
    annual_revenue: Optional[float] = None
    
    # Communication preferences
    preferred_contact_method: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = "en"
    
    # Unstructured data
    notes: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = {}
    custom_fields: Optional[Dict[str, Any]] = {}
    
    # Metadata
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}
    last_contact_date: Optional[datetime] = None
    next_follow_up: Optional[datetime] = None

class ClientCreate(ClientBase):
    pass

class ClientUpdate(BaseModel):
    # Allow partial updates
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    source: Optional[str] = None
    budget_range: Optional[str] = None
    annual_revenue: Optional[float] = None
    preferred_contact_method: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    notes: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    last_contact_date: Optional[datetime] = None
    next_follow_up: Optional[datetime] = None

class ClientResponse(ClientBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ClientSummary(BaseModel):
    """Summary view of client information."""
    id: int
    name: str
    email: str
    company: Optional[str] = None
    status: str
    priority: str
    last_contact: Optional[datetime] = None
    next_follow_up: Optional[datetime] = None
    interaction_count: int = 0
    document_count: int = 0
    tags: List[str] = []
    unstructured_fields: List[str] = []
    custom_fields: List[str] = []

# Client Interaction schemas
class ClientInteractionBase(BaseModel):
    interaction_type: str
    subject: Optional[str] = None
    content: Optional[str] = None
    outcome: Optional[str] = None
    participants: Optional[List[str]] = []
    location: Optional[str] = None
    duration_minutes: Optional[int] = None
    follow_up_required: Optional[bool] = False
    follow_up_date: Optional[datetime] = None
    follow_up_notes: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = {}
    attachments: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}

class ClientInteractionCreate(ClientInteractionBase):
    client_id: int

class ClientInteractionResponse(ClientInteractionBase):
    id: int
    client_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Client Document schemas
class ClientDocumentBase(BaseModel):
    title: str
    content: str
    document_type: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = []
    entities: Optional[List[str]] = []
    sentiment: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = "medium"
    confidential: Optional[bool] = False
    raw_data: Optional[Dict[str, Any]] = {}
    custom_fields: Optional[Dict[str, Any]] = {}
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}

class ClientDocumentCreate(ClientDocumentBase):
    client_id: int

class ClientDocumentResponse(ClientDocumentBase):
    id: int
    client_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# RAG search schemas
class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    filter_metadata: Optional[Dict[str, Any]] = None

class ClientSearchRequest(BaseModel):
    """Request schema for searching clients."""
    query: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    tags: Optional[List[str]] = None
    search_unstructured: Optional[bool] = True
    search_fields: Optional[List[str]] = None
    limit: Optional[int] = 100
    offset: Optional[int] = 0

class ClientSearchResponse(BaseModel):
    """Response schema for client search results."""
    clients: List[ClientSummary]
    total_count: int
    search_params: ClientSearchRequest

class UnstructuredDataRequest(BaseModel):
    """Request schema for adding unstructured data to clients."""
    client_id: int
    data: Dict[str, Any]
    data_type: Optional[str] = "raw_data"  # raw_data, custom_fields, metadata

class BulkClientImportRequest(BaseModel):
    """Request schema for bulk importing clients."""
    clients: List[Dict[str, Any]]
    update_existing: Optional[bool] = False
    default_status: Optional[str] = "active"
    default_priority: Optional[str] = "medium"

class ClientAnalyticsRequest(BaseModel):
    """Request schema for client analytics."""
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    group_by: Optional[str] = "status"  # status, priority, industry, source
    include_interactions: Optional[bool] = True
    include_documents: Optional[bool] = True

class ClientAnalyticsResponse(BaseModel):
    """Response schema for client analytics."""
    total_clients: int
    clients_by_status: Dict[str, int]
    clients_by_priority: Dict[str, int]
    clients_by_industry: Dict[str, int]
    clients_by_source: Dict[str, int]
    follow_up_due_count: int
    recent_interactions_count: int
    recent_documents_count: int
    unstructured_data_fields: List[str]
    custom_fields_usage: Dict[str, int]

class SearchResult(BaseModel):
    document_id: str
    content: str
    metadata: Dict[str, Any]
    distance: float

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int

# OpenAI function calling schemas
class FunctionCallRequest(BaseModel):
    function_name: str
    parameters: Dict[str, Any]
    messages: Optional[List[Dict[str, str]]] = None

class FunctionCallResponse(BaseModel):
    function_name: str
    result: Dict[str, Any]
    success: bool
