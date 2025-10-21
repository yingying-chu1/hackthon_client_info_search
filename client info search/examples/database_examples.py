#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced database
with both structured and unstructured client information.
"""

import asyncio
from datetime import datetime, timedelta
from database import (
    init_db, get_db, Client, ClientInteraction, ClientDocument,
    ClientDataHandler, DocumentDataHandler,
    get_clients_by_status, get_clients_with_follow_up_due
)
from schemas import ClientCreate, ClientInteractionCreate, ClientDocumentCreate

def example_structured_client():
    """Example of creating a client with structured data."""
    print("=== Creating Client with Structured Data ===")
    
    client_data = {
        "name": "John Smith",
        "email": "john.smith@techcorp.com",
        "phone": "+1-555-0123",
        "company": "TechCorp Inc.",
        "job_title": "CTO",
        "industry": "Technology",
        "location": "San Francisco, CA",
        "website": "https://techcorp.com",
        "status": "active",
        "priority": "high",
        "source": "referral",
        "budget_range": "$50k-100k",
        "annual_revenue": 5000000.0,
        "preferred_contact_method": "email",
        "timezone": "PST",
        "language": "en",
        "notes": "Interested in AI solutions. Very responsive to emails.",
        "tags": ["enterprise", "ai-interested", "high-value"],
        "last_contact_date": datetime.now() - timedelta(days=2),
        "next_follow_up": datetime.now() + timedelta(days=5)
    }
    
    client = ClientDataHandler.create_client_from_dict(client_data)
    print(f"Created client: {client.name} ({client.email})")
    print(f"Status: {client.status}, Priority: {client.priority}")
    print(f"Tags: {client.tags}")
    return client

def example_unstructured_client():
    """Example of creating a client with unstructured data."""
    print("\n=== Creating Client with Unstructured Data ===")
    
    # Mix of structured and unstructured data
    client_data = {
        # Structured data
        "name": "Sarah Johnson",
        "email": "sarah.j@startup.io",
        "phone": "+1-555-0456",
        "company": "StartupIO",
        "status": "prospect",
        "priority": "medium",
        
        # Unstructured data (will go to raw_data)
        "social_media": {
            "linkedin": "https://linkedin.com/in/sarahjohnson",
            "twitter": "@sarahj_startup"
        },
        "preferences": {
            "communication_style": "casual",
            "meeting_times": ["morning", "afternoon"],
            "interests": ["machine_learning", "blockchain", "sustainability"]
        },
        "previous_interactions": [
            "Met at TechCrunch conference",
            "Discussed AI implementation",
            "Sent whitepaper on ML solutions"
        ],
        "internal_notes": "Very interested in our AI platform. Mentioned budget constraints but open to phased implementation.",
        
        # Custom fields (will go to custom_fields)
        "custom_sales_stage": "qualification",
        "custom_lead_score": 85,
        "custom_referral_source": "conference_2024",
        "custom_competitor_mentioned": "CompetitorX",
        
        # Additional metadata
        "tags": ["startup", "ai-interested", "budget-conscious"],
        "metadata": {
            "lead_source_detail": "TechCrunch Disrupt 2024",
            "assigned_sales_rep": "Mike Chen",
            "territory": "West Coast"
        }
    }
    
    client = ClientDataHandler.create_client_from_dict(client_data)
    print(f"Created client: {client.name} ({client.email})")
    print(f"Raw data fields: {list(client.raw_data.keys())}")
    print(f"Custom fields: {list(client.custom_fields.keys())}")
    print(f"Metadata: {client.metadata}")
    return client

def example_client_interaction():
    """Example of creating a client interaction."""
    print("\n=== Creating Client Interaction ===")
    
    interaction_data = {
        "client_id": 1,  # Assuming client exists
        "interaction_type": "call",
        "subject": "Follow-up on AI platform demo",
        "content": "Discussed pricing options and implementation timeline. Client is very interested and wants to schedule a technical deep-dive session.",
        "outcome": "successful",
        "participants": ["John Smith", "Mike Chen (Sales Rep)"],
        "location": "phone_call",
        "duration_minutes": 45,
        "follow_up_required": True,
        "follow_up_date": datetime.now() + timedelta(days=3),
        "follow_up_notes": "Schedule technical demo with engineering team",
        "raw_data": {
            "call_recording_url": "https://storage.example.com/calls/2024-01-15-john-smith.mp3",
            "transcript_confidence": 0.95,
            "sentiment_score": 0.8,
            "key_topics": ["pricing", "implementation", "technical_requirements"]
        },
        "tags": ["follow-up", "technical-demo", "high-priority"],
        "metadata": {
            "call_quality": "excellent",
            "client_engagement": "high",
            "next_steps": "technical_demo"
        }
    }
    
    interaction = ClientInteractionCreate(**interaction_data)
    print(f"Created interaction: {interaction.interaction_type} - {interaction.subject}")
    print(f"Duration: {interaction.duration_minutes} minutes")
    print(f"Follow-up required: {interaction.follow_up_required}")
    return interaction

def example_client_document():
    """Example of creating a client document."""
    print("\n=== Creating Client Document ===")
    
    document_data = {
        "client_id": 1,  # Assuming client exists
        "title": "AI Platform Proposal - TechCorp Inc.",
        "content": """
        Executive Summary:
        This proposal outlines the implementation of our AI platform for TechCorp Inc.
        
        Key Features:
        - Machine Learning Pipeline
        - Real-time Analytics
        - Custom Model Training
        - API Integration
        
        Pricing:
        - Enterprise License: $50,000/year
        - Implementation: $25,000
        - Support: $10,000/year
        
        Timeline:
        - Phase 1: Setup and Configuration (2 weeks)
        - Phase 2: Data Integration (3 weeks)
        - Phase 3: Model Training (4 weeks)
        - Phase 4: Testing and Deployment (2 weeks)
        """,
        "document_type": "proposal",
        "file_path": "/documents/proposals/techcorp-ai-platform-2024.pdf",
        "file_size": 2048576,  # 2MB
        "mime_type": "application/pdf",
        "summary": "AI platform proposal for TechCorp including pricing and implementation timeline",
        "keywords": ["AI", "machine learning", "proposal", "TechCorp", "pricing"],
        "entities": ["TechCorp Inc.", "John Smith", "AI Platform", "Machine Learning"],
        "sentiment": "positive",
        "category": "sales",
        "subcategory": "proposal",
        "priority": "high",
        "confidential": True,
        "raw_data": {
            "proposal_version": "2.1",
            "last_reviewed_by": "Mike Chen",
            "approval_status": "pending",
            "competitor_analysis": {
                "competitor_a": "more_expensive",
                "competitor_b": "less_features"
            }
        },
        "custom_fields": {
            "custom_proposal_id": "PROP-2024-001",
            "custom_approval_required": True,
            "custom_legal_review": "pending"
        },
        "tags": ["proposal", "ai-platform", "techcorp", "high-value"],
        "metadata": {
            "created_by": "Mike Chen",
            "reviewed_by": "Sarah Wilson",
            "approval_deadline": "2024-02-01"
        }
    }
    
    document = ClientDocumentCreate(**document_data)
    print(f"Created document: {document.title}")
    print(f"Type: {document.document_type}, Priority: {document.priority}")
    print(f"Confidential: {document.confidential}")
    print(f"Keywords: {document.keywords}")
    return document

def example_search_unstructured_data():
    """Example of searching unstructured data."""
    print("\n=== Searching Unstructured Data ===")
    
    # This would typically be done with a database session
    # For demonstration purposes, we'll show the concept
    
    search_examples = [
        {
            "search_term": "AI",
            "description": "Search for clients interested in AI"
        },
        {
            "search_term": "startup",
            "description": "Search for startup clients"
        },
        {
            "search_term": "budget",
            "description": "Search for budget-related information"
        }
    ]
    
    for search in search_examples:
        print(f"Search: '{search['search_term']}' - {search['description']}")
        print("  Would search in: raw_data, custom_fields, notes")
        print("  Could find matches in unstructured fields like:")
        print("    - preferences.interests")
        print("    - internal_notes")
        print("    - custom_competitor_mentioned")
        print()

def example_analytics():
    """Example of client analytics."""
    print("\n=== Client Analytics Example ===")
    
    analytics_data = {
        "total_clients": 150,
        "clients_by_status": {
            "active": 45,
            "prospect": 30,
            "lead": 25,
            "inactive": 50
        },
        "clients_by_priority": {
            "high": 20,
            "medium": 80,
            "low": 50
        },
        "clients_by_industry": {
            "Technology": 60,
            "Healthcare": 25,
            "Finance": 20,
            "Manufacturing": 15,
            "Other": 30
        },
        "follow_up_due_count": 12,
        "recent_interactions_count": 45,
        "recent_documents_count": 23,
        "unstructured_data_fields": [
            "social_media",
            "preferences",
            "previous_interactions",
            "internal_notes"
        ],
        "custom_fields_usage": {
            "custom_sales_stage": 120,
            "custom_lead_score": 95,
            "custom_referral_source": 80,
            "custom_competitor_mentioned": 45
        }
    }
    
    print("Analytics Summary:")
    print(f"Total Clients: {analytics_data['total_clients']}")
    print(f"Active Clients: {analytics_data['clients_by_status']['active']}")
    print(f"Follow-ups Due: {analytics_data['follow_up_due_count']}")
    print(f"Most Common Industry: Technology ({analytics_data['clients_by_industry']['Technology']} clients)")
    print(f"Custom Fields Usage: {len(analytics_data['custom_fields_usage'])} different fields")

def main():
    """Main function to run all examples."""
    print("Client Information Database Examples")
    print("=" * 50)
    
    # Initialize database
    init_db()
    print("Database initialized")
    
    # Run examples
    example_structured_client()
    example_unstructured_client()
    example_client_interaction()
    example_client_document()
    example_search_unstructured_data()
    example_analytics()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nKey Benefits of this Enhanced Database:")
    print("1. Structured fields for common client data")
    print("2. Flexible unstructured data storage in raw_data")
    print("3. Custom fields for specific use cases")
    print("4. Rich metadata and tagging system")
    print("5. Relationship tracking (interactions, documents)")
    print("6. Search capabilities across all data types")
    print("7. Analytics and reporting features")

if __name__ == "__main__":
    main()
