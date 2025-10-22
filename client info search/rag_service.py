"""
Chroma RAG service for vector search and document management.
"""

import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import os
import asyncio
from schemas import SearchResult

class RAGService:
    """RAG service using Chroma for vector search."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "documents"
        
    async def initialize(self):
        """Initialize Chroma client and collection."""
        try:
            # Initialize Chroma client with proper persistence
            import os
            persist_dir = "./chroma_db"
            os.makedirs(persist_dir, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document collection for RAG search"}
                )
                
            print(f"RAG service initialized with collection: {self.collection_name}")
            
        except Exception as e:
            print(f"Error initializing RAG service: {e}")
            raise
    
    async def add_document(
        self, 
        document_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document to the vector store."""
        try:
            # Generate unique ID if not provided
            if not document_id:
                document_id = str(uuid.uuid4())
            
            # Add document to collection
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            print(f"Added document {document_id} to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding document {document_id}: {e}")
            return False
    
    async def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        try:
            # Prepare where clause for filtering
            where_clause = filter_metadata if filter_metadata else None
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
            
            # Convert results to SearchResult objects
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    search_results.append(SearchResult(
                        document_id=results['ids'][0][i],
                        content=doc,
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                        distance=results['distances'][0][i] if results['distances'] else 0.0
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def update_document(
        self, 
        document_id: str, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update an existing document in the vector store."""
        try:
            # Update document in collection
            self.collection.update(
                documents=[content],
                metadatas=[metadata],
                ids=[document_id]
            )
            
            print(f"Updated document {document_id} in vector store")
            return True
            
        except Exception as e:
            print(f"Error updating document {document_id}: {e}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            # Delete document from collection
            self.collection.delete(ids=[document_id])
            
            print(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            # Chroma client doesn't need explicit cleanup
            pass
        print("RAG service cleaned up")
