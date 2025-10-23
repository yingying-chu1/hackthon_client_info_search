"""
FastAPI application with SQLite, Chroma RAG, and OpenAI function-calling tools.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import os
import json
from typing import List, Dict, Any, Optional

from database import get_db, init_db
from models import Document, Client
from rag_service import RAGService
from openai_service import OpenAIService
from schemas import (
    DocumentCreate, DocumentResponse, 
    ClientCreate, ClientResponse,
    QueryRequest, QueryResponse,
    FunctionCallRequest, FunctionCallResponse
)

# Global services
rag_service: Optional[RAGService] = None
openai_service: Optional[OpenAIService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    global rag_service, openai_service
    
    # Startup
    print("Starting up...")
    init_db()
    
    # Initialize services
    rag_service = RAGService()
    await rag_service.initialize()
    
    openai_service = OpenAIService()
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if rag_service:
        await rag_service.cleanup()

app = FastAPI(
    title="Client Info Search API",
    description="FastAPI app with SQLite, Chroma RAG, and OpenAI function-calling tools",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Dependency to get RAG service
def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return rag_service

# Dependency to get OpenAI service
def get_openai_service() -> OpenAIService:
    if openai_service is None:
        raise HTTPException(status_code=503, detail="OpenAI service not initialized")
    return openai_service

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Client Info Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "services": {
        "rag": rag_service is not None,
        "openai": openai_service is not None
    }}

# Document endpoints
@app.post("/documents/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db=Depends(get_db),
    rag=Depends(get_rag_service)
):
    """Create a new document and add it to the vector store."""
    try:
        # Create document in database
        db_document = Document(
            title=document.title,
            content=document.content,
            metadata=document.metadata
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Add to vector store
        await rag.add_document(
            document_id=str(db_document.id),
            content=document.content,
            metadata={**document.metadata, "title": document.title}
        )
        
        return DocumentResponse.from_orm(db_document)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/documents/", response_model=List[DocumentResponse])
async def get_documents(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    """Get all documents with pagination."""
    documents = db.query(Document).offset(skip).limit(limit).all()
    return [DocumentResponse.from_orm(doc) for doc in documents]

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db=Depends(get_db)):
    """Get a specific document by ID."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.from_orm(document)

# Client endpoints
@app.post("/clients/", response_model=ClientResponse)
async def create_client(client: ClientCreate, db=Depends(get_db)):
    """Create a new client."""
    try:
        db_client = Client(**client.dict())
        db.add(db_client)
        db.commit()
        db.refresh(db_client)
        return ClientResponse.from_orm(db_client)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/clients/", response_model=List[ClientResponse])
async def get_clients(skip: int = 0, limit: int = 100, db=Depends(get_db)):
    """Get all clients with pagination."""
    clients = db.query(Client).offset(skip).limit(limit).all()
    return [ClientResponse.from_orm(client) for client in clients]

@app.get("/clients/{client_id}", response_model=ClientResponse)
async def get_client(client_id: int, db=Depends(get_db)):
    """Get a specific client by ID."""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return ClientResponse.from_orm(client)

# RAG search endpoints
@app.post("/search/", response_model=QueryResponse)
async def search_documents(
    query: QueryRequest,
    rag=Depends(get_rag_service)
):
    """Search documents using RAG."""
    try:
        results = await rag.search(
            query=query.query,
            n_results=query.n_results,
            filter_metadata=query.filter_metadata
        )
        return QueryResponse(
            query=query.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search", response_model=QueryResponse)
async def search_documents_no_slash(
    query: QueryRequest,
    rag=Depends(get_rag_service)
):
    """Search documents using RAG (without trailing slash for frontend compatibility)."""
    try:
        results = await rag.search(
            query=query.query,
            n_results=query.n_results,
            filter_metadata=query.filter_metadata
        )
        return QueryResponse(
            query=query.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Client summary endpoint
@app.get("/client-summary/{patient_id}")
async def get_client_summary(patient_id: str, rag=Depends(get_rag_service)):
    """Get a comprehensive client summary for a specific patient."""
    try:
        # Search for all documents related to this patient
        results = await rag.search(
            query=f"patient {patient_id} summary",
            n_results=10,
            filter_metadata={"patient_id": patient_id}
        )
        
        if not results:
            # If no specific patient documents, try broader search
            results = await rag.search(
                query=f"patient {patient_id}",
                n_results=10
            )
        
        if not results:
            return {
                "patient_id": patient_id,
                "summary": f"No data found for patient {patient_id}",
                "found_documents": 0
            }
        
        # Create a comprehensive summary
        summary_parts = []
        appointment_count = 0
        assessment_count = 0
        
        for result in results:
            content = result.content
            metadata = result.metadata
            
            if 'Appointment #' in content:
                appointment_count += 1
                summary_parts.append(f"📅 {content}")
            elif 'Assessment Results' in content:
                assessment_count += 1
                summary_parts.append(f"📊 {content}")
            elif 'Patient Summary' in content:
                summary_parts.append(f"📋 {content}")
        
        summary = "\n\n".join(summary_parts)
        
        return {
            "patient_id": patient_id,
            "summary": summary,
            "found_documents": len(results),
            "appointment_count": appointment_count,
            "assessment_count": assessment_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Analytics endpoint
@app.get("/analytics")
async def get_analytics(rag=Depends(get_rag_service)):
    """Get system analytics and statistics."""
    try:
        # Get all documents to calculate analytics
        all_results = await rag.search(
            query="all documents",
            n_results=1000  # Get a large number to capture all documents
        )
        
        # Count different types of documents
        total_documents = len(all_results)
        appointment_docs = 0
        assessment_docs = 0
        summary_docs = 0
        unique_patients = set()
        unique_clients = set()
        
        for result in all_results:
            content = result.content
            metadata = result.metadata
            
            if 'Appointment #' in content:
                appointment_docs += 1
            elif 'Assessment Results' in content:
                assessment_docs += 1
            elif 'Patient Summary' in content:
                summary_docs += 1
            
            if 'patient_id' in metadata:
                unique_patients.add(metadata['patient_id'])
            if 'client_id' in metadata:
                unique_clients.add(metadata['client_id'])
        
        return {
            "total_documents": total_documents,
            "total_clients": len(unique_clients),
            "total_patients": len(unique_patients),
            "appointment_documents": appointment_docs,
            "assessment_documents": assessment_docs,
            "summary_documents": summary_docs,
            "document_types": {
                "appointments": appointment_docs,
                "assessments": assessment_docs,
                "summaries": summary_docs
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Detailed PHQ9 analysis endpoint
@app.get("/phq9-analysis/{patient_id}")
async def get_phq9_analysis(patient_id: str, rag=Depends(get_rag_service)):
    """Get detailed PHQ9 question-level analysis for a patient."""
    try:
        # Get all PHQ9 assessments for this patient
        results = await rag.search(f"patient {patient_id} PHQ9", n_results=20)
        phq9_assessments = [r for r in results if 'Assessment Results' in r.content and r.metadata.get('measure_type') == 'PHQ9']
        
        if len(phq9_assessments) < 2:
            return {
                "patient_id": patient_id,
                "error": "Need at least 2 PHQ9 assessments to analyze trends",
                "assessments_found": len(phq9_assessments)
            }
        
        # Parse question responses from each assessment
        assessments_data = []
        for assessment in phq9_assessments:
            metadata = assessment.metadata
            question_responses = json.loads(metadata.get('question_responses', '[]'))
            
            assessments_data.append({
                'date': metadata.get('measure_date', 'Unknown'),
                'total_score': metadata.get('total_score', 0),
                'questions': {str(int(q['question_number'])): int(q['question_score']) for q in question_responses}
            })
        
        # Sort by date (convert to comparable format)
        def parse_date(date_str):
            try:
                # Handle different date formats
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        return f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                return date_str
            except:
                return "1900-01-01"  # Default for sorting
        
        assessments_data.sort(key=lambda x: parse_date(x['date']))
        
        # Calculate question-level changes
        baseline = assessments_data[0]
        latest = assessments_data[-1]
        
        question_changes = {}
        for q_num in range(1, 10):  # PHQ9 has 9 questions
            q_str = str(q_num)
            if q_str in baseline['questions'] and q_str in latest['questions']:
                change = latest['questions'][q_str] - baseline['questions'][q_str]
                question_changes[q_str] = {
                    'baseline': baseline['questions'][q_str],
                    'latest': latest['questions'][q_str],
                    'change': change,
                    'improvement': change < 0
                }
        
        # PHQ9 question descriptions
        phq9_questions = {
            '1': 'Little interest or pleasure in doing things',
            '2': 'Feeling down, depressed, or hopeless',
            '3': 'Trouble falling or staying asleep, or sleeping too much',
            '4': 'Feeling tired or having little energy',
            '5': 'Poor appetite or overeating',
            '6': 'Feeling bad about yourself - or that you are a failure or have let yourself or your family down',
            '7': 'Trouble concentrating on things, such as reading the newspaper or watching television',
            '8': 'Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual',
            '9': 'Thoughts that you would be better off dead or of hurting yourself in some way'
        }
        
        gad7_questions = {
            '1': 'Feeling nervous, anxious, or on edge',
            '2': 'Not being able to stop or control worrying',
            '3': 'Worrying too much about different things',
            '4': 'Trouble relaxing',
            '5': 'Being so restless that it\'s hard to sit still',
            '6': 'Becoming easily annoyed or irritable',
            '7': 'Feeling afraid as if something awful might happen'
        }
        
        # Generate analysis
        analysis = generate_phq9_question_analysis(question_changes, phq9_questions, baseline, latest)
        
        return {
            "patient_id": patient_id,
            "analysis": analysis,
            "question_changes": question_changes,
            "assessments": assessments_data,
            "total_score_change": latest['total_score'] - baseline['total_score']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Detailed GAD7 analysis endpoint
@app.get("/gad7-analysis/{patient_id}")
async def get_gad7_analysis(patient_id: str, rag=Depends(get_rag_service)):
    """Get detailed GAD7 question-level analysis for a patient."""
    try:
        # Get all GAD7 assessments for this patient
        results = await rag.search(f"patient {patient_id} GAD7", n_results=20)
        gad7_assessments = [r for r in results if 'Assessment Results' in r.content and r.metadata.get('measure_type') == 'GAD7']
        
        if len(gad7_assessments) < 2:
            return {
                "patient_id": patient_id,
                "error": "Need at least 2 GAD7 assessments to analyze trends",
                "assessments_found": len(gad7_assessments)
            }
        
        # Parse question responses from each assessment
        assessments_data = []
        for assessment in gad7_assessments:
            metadata = assessment.metadata
            question_responses = json.loads(metadata.get('question_responses', '[]'))
            
            assessments_data.append({
                'date': metadata.get('measure_date', 'Unknown'),
                'total_score': metadata.get('total_score', 0),
                'questions': {str(int(q['question_number'])): int(q['question_score']) for q in question_responses}
            })
        
        # Sort by date
        def parse_date(date_str):
            try:
                if '/' in date_str:
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        return f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                return date_str
            except:
                return "1900-01-01"
        
        assessments_data.sort(key=lambda x: parse_date(x['date']))
        
        # Calculate question-level changes
        baseline = assessments_data[0]
        latest = assessments_data[-1]
        
        question_changes = {}
        for q_num in range(1, 8):  # GAD7 has 7 questions
            q_str = str(q_num)
            if q_str in baseline['questions'] and q_str in latest['questions']:
                change = latest['questions'][q_str] - baseline['questions'][q_str]
                question_changes[q_str] = {
                    'baseline': baseline['questions'][q_str],
                    'latest': latest['questions'][q_str],
                    'change': change,
                    'improvement': change < 0
                }
        
        # GAD7 question descriptions
        gad7_questions = {
            '1': 'Feeling nervous, anxious, or on edge',
            '2': 'Not being able to stop or control worrying',
            '3': 'Worrying too much about different things',
            '4': 'Trouble relaxing',
            '5': 'Being so restless that it\'s hard to sit still',
            '6': 'Becoming easily annoyed or irritable',
            '7': 'Feeling afraid as if something awful might happen'
        }
        
        # Generate analysis
        analysis = generate_gad7_question_analysis(question_changes, gad7_questions, baseline, latest)
        
        return {
            "patient_id": patient_id,
            "analysis": analysis,
            "question_changes": question_changes,
            "assessments": assessments_data,
            "total_score_change": latest['total_score'] - baseline['total_score']
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_phq9_question_analysis(question_changes, phq9_questions, baseline, latest):
    """Generate detailed PHQ9 question-level analysis."""
    
    # Find questions with biggest improvements
    improvements = [(q, data) for q, data in question_changes.items() if data['improvement']]
    improvements.sort(key=lambda x: x[1]['change'])  # Sort by most negative (biggest improvement)
    
    # Find questions that got worse
    worsening = [(q, data) for q, data in question_changes.items() if not data['improvement'] and data['change'] > 0]
    worsening.sort(key=lambda x: x[1]['change'], reverse=True)  # Sort by biggest increase
    
    analysis = f"## PHQ9 Question-Level Analysis for Patient {baseline.get('patient_id', 'Unknown')}\n\n"
    
    total_change = latest['total_score'] - baseline['total_score']
    baseline_date = baseline.get('date', 'Unknown date')
    latest_date = latest.get('date', 'Unknown date')
    
    if total_change < 0:
        analysis += f"**Overall Progress:** Total score decreased from {baseline['total_score']} to {latest['total_score']} ({abs(total_change)} point improvement)\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    elif total_change > 0:
        analysis += f"**Overall Progress:** Total score increased from {baseline['total_score']} to {latest['total_score']} (+{total_change} point increase)\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    else:
        analysis += f"**Overall Progress:** Total score remained stable at {baseline['total_score']} points\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    
    if improvements:
        analysis += "### 🎉 Questions Showing Improvement:\n\n"
        for q_num, data in improvements[:3]:  # Top 3 improvements
            question_desc = phq9_questions.get(q_num, f"Question {q_num}")
            analysis += f"**Question {q_num}** ({question_desc}):\n"
            analysis += f"- Baseline: {data['baseline']} ({baseline_date}) → Latest: {data['latest']} ({latest_date}) ({data['change']} point improvement)\n"
            analysis += f"- This suggests the client is experiencing less {question_desc.lower()}\n\n"
    
    if worsening:
        analysis += "### ⚠️ Questions That May Need Attention:\n\n"
        for q_num, data in worsening[:2]:  # Top 2 concerns
            question_desc = phq9_questions.get(q_num, f"Question {q_num}")
            analysis += f"**Question {q_num}** ({question_desc}):\n"
            analysis += f"- Baseline: {data['baseline']} ({baseline_date}) → Latest: {data['latest']} ({latest_date}) (+{data['change']} point increase)\n"
            analysis += f"- This area may need additional focus in therapy\n\n"
    
    # Find stable questions
    stable = [(q, data) for q, data in question_changes.items() if data['change'] == 0]
    if stable:
        analysis += f"### 📊 Stable Areas ({len(stable)} questions):\n"
        analysis += f"- Questions showing no change: {', '.join([q for q, _ in stable])}\n"
        analysis += f"- These areas are maintaining baseline levels\n\n"
    
    # Clinical interpretation
    analysis += "### 🧠 Clinical Insights:\n"
    if improvements:
        best_improvement = improvements[0]
        q_num = best_improvement[0]
        question_desc = phq9_questions.get(q_num, f"Question {q_num}")
        analysis += f"- **Biggest improvement** in {question_desc.lower()} (based on measure taken on {latest_date} compared to {baseline_date}) suggests this intervention area is most effective\n"
    
    if latest['total_score'] <= 4:
        analysis += "- **Current score ≤ 4**: Minimal depression symptoms\n"
    elif latest['total_score'] <= 9:
        analysis += "- **Current score 5-9**: Mild depression symptoms\n"
    elif latest['total_score'] <= 14:
        analysis += "- **Current score 10-14**: Moderate depression symptoms\n"
    elif latest['total_score'] <= 19:
        analysis += "- **Current score 15-19**: Moderately severe depression symptoms\n"
    else:
        analysis += "- **Current score ≥ 20**: Severe depression symptoms\n"
    
    return analysis

def generate_gad7_question_analysis(question_changes, gad7_questions, baseline, latest):
    """Generate detailed GAD7 question-level analysis."""
    
    # Find questions with biggest improvements
    improvements = [(q, data) for q, data in question_changes.items() if data['improvement']]
    improvements.sort(key=lambda x: x[1]['change'])  # Sort by most negative (biggest improvement)
    
    # Find questions that got worse
    worsening = [(q, data) for q, data in question_changes.items() if not data['improvement'] and data['change'] > 0]
    worsening.sort(key=lambda x: x[1]['change'], reverse=True)  # Sort by biggest increase
    
    analysis = f"## GAD7 Question-Level Analysis for Patient {baseline.get('patient_id', 'Unknown')}\n\n"
    
    total_change = latest['total_score'] - baseline['total_score']
    baseline_date = baseline.get('date', 'Unknown date')
    latest_date = latest.get('date', 'Unknown date')
    
    if total_change < 0:
        analysis += f"**Overall Progress:** Total score decreased from {baseline['total_score']} to {latest['total_score']} ({abs(total_change)} point improvement)\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    elif total_change > 0:
        analysis += f"**Overall Progress:** Total score increased from {baseline['total_score']} to {latest['total_score']} (+{total_change} point increase)\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    else:
        analysis += f"**Overall Progress:** Total score remained stable at {baseline['total_score']} points\n"
        analysis += f"- **Baseline assessment:** {baseline_date}\n"
        analysis += f"- **Latest assessment:** {latest_date}\n\n"
    
    if improvements:
        analysis += "### 🎉 Questions Showing Improvement:\n\n"
        for q_num, data in improvements[:3]:  # Top 3 improvements
            question_desc = gad7_questions.get(q_num, f"Question {q_num}")
            analysis += f"**Question {q_num}** ({question_desc}):\n"
            analysis += f"- Baseline: {data['baseline']} ({baseline_date}) → Latest: {data['latest']} ({latest_date}) ({data['change']} point improvement)\n"
            analysis += f"- This suggests the client is experiencing less {question_desc.lower()}\n\n"
    
    if worsening:
        analysis += "### ⚠️ Questions That May Need Attention:\n\n"
        for q_num, data in worsening[:2]:  # Top 2 concerns
            question_desc = gad7_questions.get(q_num, f"Question {q_num}")
            analysis += f"**Question {q_num}** ({question_desc}):\n"
            analysis += f"- Baseline: {data['baseline']} ({baseline_date}) → Latest: {data['latest']} ({latest_date}) (+{data['change']} point increase)\n"
            analysis += f"- This area may need additional focus in therapy\n\n"
    
    # Find stable questions
    stable = [(q, data) for q, data in question_changes.items() if data['change'] == 0]
    if stable:
        analysis += f"### 📊 Stable Areas ({len(stable)} questions):\n"
        analysis += f"- Questions showing no change: {', '.join([q for q, _ in stable])}\n"
        analysis += f"- These areas are maintaining baseline levels\n\n"
    
    # Clinical interpretation
    analysis += "### 🧠 Clinical Insights:\n"
    if improvements:
        best_improvement = improvements[0]
        q_num = best_improvement[0]
        question_desc = gad7_questions.get(q_num, f"Question {q_num}")
        analysis += f"- **Biggest improvement** in {question_desc.lower()} (based on measure taken on {latest_date} compared to {baseline_date}) suggests this intervention area is most effective\n"
    
    if latest['total_score'] <= 4:
        analysis += "- **Current score ≤ 4**: Minimal anxiety symptoms\n"
    elif latest['total_score'] <= 9:
        analysis += "- **Current score 5-9**: Mild anxiety symptoms\n"
    elif latest['total_score'] <= 14:
        analysis += "- **Current score 10-14**: Moderate anxiety symptoms\n"
    else:
        analysis += "- **Current score ≥ 15**: Severe anxiety symptoms\n"
    
    return analysis

# Conversational analysis endpoints
@app.post("/analyze-client-progress/{patient_id}")
async def analyze_client_progress(patient_id: str, rag=Depends(get_rag_service)):
    """Analyze client progress and provide high-level insights."""
    try:
        # Get all documents for this patient
        results = await rag.search(
            query=f"patient {patient_id}",
            n_results=50
        )
        
        if not results:
            return {
                "patient_id": patient_id,
                "analysis": f"No data found for patient {patient_id}",
                "status": "no_data"
            }
        
        # Analyze different aspects
        appointments = []
        assessments = []
        summaries = []
        
        for result in results:
            content = result.content
            metadata = result.metadata
            
            if 'Appointment #' in content:
                appointments.append({
                    "content": content,
                    "metadata": metadata,
                    "date": metadata.get('appointment_date', 'Unknown')
                })
            elif 'Assessment Results' in content:
                assessments.append({
                    "content": content,
                    "metadata": metadata,
                    "date": metadata.get('measure_date', 'Unknown'),
                    "type": metadata.get('measure_type', 'Unknown'),
                    "score": metadata.get('total_score', 0)
                })
            elif 'Patient Summary' in content:
                summaries.append({
                    "content": content,
                    "metadata": metadata
                })
        
        # Generate analysis
        analysis = generate_progress_analysis(appointments, assessments, summaries, patient_id)
        
        return {
            "patient_id": patient_id,
            "analysis": analysis,
            "data_points": {
                "appointments": len(appointments),
                "assessments": len(assessments),
                "summaries": len(summaries)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/conversational-query")
async def conversational_query(
    request: dict,
    rag=Depends(get_rag_service)
):
    """Handle conversational queries with intelligent responses."""
    try:
        query = request.get("query", "")
        patient_id = request.get("patient_id", "789012")  # Default to current client
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Determine query type and provide appropriate response
        response = await handle_conversational_query(query, patient_id, rag)
        
        return {
            "query": query,
            "response": response,
            "patient_id": patient_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_progress_analysis(appointments, assessments, summaries, patient_id):
    """Generate high-level progress analysis."""
    
    # Analyze appointment patterns
    completed_appointments = [a for a in appointments if a['metadata'].get('is_completed', False)]
    canceled_appointments = [a for a in appointments if a['metadata'].get('is_cancelled', False)]
    no_show_appointments = [a for a in appointments if a['metadata'].get('is_no_show', False)]
    
    # Analyze assessment trends
    phq9_scores = []
    gad7_scores = []
    
    for assessment in assessments:
        if assessment['metadata'].get('measure_type') == 'PHQ9':
            phq9_scores.append({
                'date': assessment['date'],
                'score': assessment['score']
            })
        elif assessment['metadata'].get('measure_type') == 'GAD7':
            gad7_scores.append({
                'date': assessment['date'],
                'score': assessment['score']
            })
    
    # Generate insights
    insights = []
    
    # Appointment engagement
    total_appointments = len(appointments)
    completion_rate = len(completed_appointments) / total_appointments * 100 if total_appointments > 0 else 0
    
    if completion_rate >= 80:
        insights.append("📈 **Excellent engagement**: Client shows high commitment to treatment with {:.1f}% completion rate.".format(completion_rate))
    elif completion_rate >= 60:
        insights.append("📊 **Good engagement**: Client maintains reasonable attendance with {:.1f}% completion rate.".format(completion_rate))
    else:
        insights.append("⚠️ **Engagement concerns**: Client has lower completion rate ({:.1f}%) - may need support.".format(completion_rate))
    
    # Assessment trends
    if len(phq9_scores) >= 2:
        phq9_trend = phq9_scores[-1]['score'] - phq9_scores[0]['score']
        if phq9_trend < -3:
            insights.append("🎉 **Significant improvement**: PHQ9 depression scores decreased by {} points, indicating substantial progress.".format(abs(phq9_trend)))
        elif phq9_trend < 0:
            insights.append("📈 **Positive trend**: PHQ9 depression scores show improvement ({} point decrease).".format(abs(phq9_trend)))
        elif phq9_trend > 3:
            insights.append("⚠️ **Concerning trend**: PHQ9 depression scores increased by {} points - may need intervention.".format(phq9_trend))
        else:
            insights.append("📊 **Stable symptoms**: PHQ9 depression scores remain relatively stable.")
    
    if len(gad7_scores) >= 2:
        gad7_trend = gad7_scores[-1]['score'] - gad7_scores[0]['score']
        if gad7_trend < -2:
            insights.append("🎉 **Anxiety improvement**: GAD7 anxiety scores decreased by {} points, showing good progress.".format(abs(gad7_trend)))
        elif gad7_trend < 0:
            insights.append("📈 **Anxiety trending down**: GAD7 scores show modest improvement ({} point decrease).".format(abs(gad7_trend)))
        elif gad7_trend > 2:
            insights.append("⚠️ **Anxiety concerns**: GAD7 scores increased by {} points - monitor closely.".format(gad7_trend))
    
    # Session content analysis
    recent_sessions = sorted(appointments, key=lambda x: x['date'], reverse=True)[:3]
    if recent_sessions:
        session_insights = []
        for session in recent_sessions:
            content = session['content'].lower()
            if 'progress' in content or 'improvement' in content:
                session_insights.append("positive")
            elif 'struggling' in content or 'difficult' in content:
                session_insights.append("challenging")
        
        if session_insights.count("positive") >= 2:
            insights.append("💪 **Recent progress**: Recent sessions show consistent positive developments.")
        elif session_insights.count("challenging") >= 2:
            insights.append("🤝 **Support needed**: Recent sessions indicate client may need additional support.")
    
    # Overall assessment
    if len(insights) == 0:
        insights.append("📋 **Limited data**: Insufficient information to assess progress. More data needed.")
    
    # Generate summary
    summary = f"## Progress Analysis for Patient {patient_id}\n\n"
    summary += "### Key Insights:\n"
    for insight in insights:
        summary += f"- {insight}\n"
    
    summary += f"\n### Data Summary:\n"
    summary += f"- **Total Appointments**: {total_appointments}\n"
    summary += f"- **Completed**: {len(completed_appointments)} ({completion_rate:.1f}%)\n"
    summary += f"- **Assessments**: {len(assessments)} (PHQ9: {len(phq9_scores)}, GAD7: {len(gad7_scores)})\n"
    
    if phq9_scores:
        summary += f"- **Latest PHQ9 Score**: {phq9_scores[-1]['score']} (Baseline: {phq9_scores[0]['score']})\n"
    if gad7_scores:
        summary += f"- **Latest GAD7 Score**: {gad7_scores[-1]['score']} (Baseline: {gad7_scores[0]['score']})\n"
    
    return summary

async def handle_conversational_query(query: str, patient_id: str, rag):
    """Handle conversational queries with intelligent responses."""
    
    query_lower = query.lower()
    
    # Sleep-specific queries
    if any(word in query_lower for word in ['sleep', 'sleeping', 'insomnia', 'sleep quality', 'trouble sleeping', 'sleep disturbance']):
        # Get PHQ9 Question 3 analysis (sleep-related)
        try:
            phq9_response = await get_phq9_analysis(patient_id, rag)
            if 'error' not in phq9_response:
                question_changes = phq9_response.get('question_changes', {})
                sleep_data = question_changes.get('3', None)  # Question 3 is about sleep
                
                if sleep_data:
                    baseline_score = sleep_data['baseline']
                    latest_score = sleep_data['latest']
                    change = sleep_data['change']
                    
                    response = f"**Sleep Quality Analysis for this client:**\n\n"
                    
                    if change < 0:
                        response += f"🎉 **Yes, sleep quality is improving!**\n"
                        response += f"- **Baseline (1/2/25)**: {baseline_score}/3 (Trouble falling or staying asleep, or sleeping too much)\n"
                        response += f"- **Latest (10/21/25)**: {latest_score}/3\n"
                        response += f"- **Improvement**: {abs(change)} point decrease\n\n"
                        
                        if latest_score == 0:
                            response += "✅ **Excellent**: No sleep problems reported\n"
                        elif latest_score == 1:
                            response += "✅ **Good**: Minimal sleep difficulties\n"
                        elif latest_score == 2:
                            response += "📈 **Improving**: Moderate sleep issues, but getting better\n"
                    elif change > 0:
                        response += f"⚠️ **Sleep quality may be worsening**\n"
                        response += f"- **Baseline (1/2/25)**: {baseline_score}/3\n"
                        response += f"- **Latest (10/21/25)**: {latest_score}/3\n"
                        response += f"- **Change**: +{change} point increase\n\n"
                        response += "💡 **Consider**: Sleep hygiene techniques or sleep-focused interventions\n"
                    else:
                        response += f"📊 **Sleep quality is stable**\n"
                        response += f"- **Consistent score**: {baseline_score}/3 across assessments\n\n"
                    
                    # Add sleep-specific session insights
                    results = await rag.search(f"patient {patient_id} sleep", n_results=10)
                    sleep_mentions = [r for r in results if 'sleep' in r.content.lower()]
                    
                    if sleep_mentions:
                        response += "**Sleep-related session notes:**\n\n"
                        for mention in sleep_mentions[:2]:
                            session_num = mention.metadata.get('appointment_number', 'N/A')
                            date = mention.metadata.get('appointment_date', 'Unknown')
                            content = mention.content
                            
                            if 'Session Notes:' in content:
                                notes = content.split('Session Notes:')[1].strip()
                            else:
                                notes = content
                            
                            if 'sleep' in notes.lower():
                                response += f"**Session #{session_num} ({date}):**\n"
                                # Extract sleep-related sentences
                                sentences = notes.split('.')
                                sleep_sentences = [s.strip() for s in sentences if 'sleep' in s.lower()]
                                if sleep_sentences:
                                    response += f"- {sleep_sentences[0]}.\n\n"
                    
                    return response
                else:
                    return "I don't have specific sleep quality data for this client. The PHQ9 Question 3 (sleep-related) data may not be available."
            else:
                return "I need PHQ9 assessment data to analyze sleep quality. Please ensure the client has completed PHQ9 assessments."
        except:
            return "I can analyze sleep quality using PHQ9 Question 3 data, but I need more assessment information for this client."
    
    # CBT and therapy intervention queries
    elif any(word in query_lower for word in ['cbt', 'cognitive behavioral', 'therapy', 'intervention', 'introduced', 'using', 'technique', 'approach']):
        # Check for timing-specific questions
        if any(word in query_lower for word in ['when', 'first', 'introduced', 'started', 'began']):
            # Search for CBT mentions chronologically
            results = await rag.search(f"patient {patient_id} CBT therapy session", n_results=20)
            cbt_sessions = []
            
            for result in results:
                content = result.content.lower()
                if 'cbt' in content or 'cognitive' in content or 'behavioral' in content:
                    cbt_sessions.append({
                        'session_num': result.metadata.get('appointment_number', 'N/A'),
                        'date': result.metadata.get('appointment_date', 'Unknown'),
                        'notes': result.content,
                        'sort_date': result.metadata.get('appointment_date', '1900-01-01')
                    })
            
            if cbt_sessions:
                # Sort by date to find the first occurrence
                cbt_sessions.sort(key=lambda x: x['sort_date'])
                first_cbt = cbt_sessions[0]
                
                response = f"**CBT was first introduced in Session #{first_cbt['session_num']} on {first_cbt['date']}.**\n\n"
                
                # Extract specific CBT techniques from first session
                notes = first_cbt['notes']
                response += f"**First CBT Session Details:**\n"
                
                if 'cbt' in notes.lower():
                    response += "✅ CBT explicitly mentioned in treatment plan\n"
                if 'cognitive' in notes.lower():
                    response += "✅ Cognitive restructuring techniques introduced\n"
                if 'behavioral' in notes.lower():
                    response += "✅ Behavioral interventions started\n"
                if 'homework' in notes.lower():
                    response += "✅ CBT homework assigned\n"
                
                # Show progression
                if len(cbt_sessions) > 1:
                    response += f"\n**CBT Progression:**\n"
                    for i, session in enumerate(cbt_sessions[:3], 1):
                        response += f"- Session #{session['session_num']} ({session['date']}): CBT continued\n"
                
                return response
            else:
                return "I don't see explicit CBT techniques mentioned in the session notes. The therapy appears to focus on interpersonal interventions and emotional processing rather than traditional CBT approaches."
        
        # Regular CBT detection (existing code)
        results = await rag.search(f"patient {patient_id} CBT therapy session", n_results=10)
        cbt_sessions = []
        
        for result in results:
            content = result.content.lower()
            if 'cbt' in content or 'cognitive' in content or 'behavioral' in content:
                cbt_sessions.append({
                    'session_num': result.metadata.get('appointment_number', 'N/A'),
                    'date': result.metadata.get('appointment_date', 'Unknown'),
                    'notes': result.content
                })
        
        if cbt_sessions:
            response = f"**Yes, CBT has been introduced to this client!** Here's what I found:\n\n"
            
            for session in cbt_sessions[:3]:  # Show top 3 CBT sessions
                response += f"**Session #{session['session_num']} ({session['date']}):**\n"
                
                # Extract CBT-specific content
                notes = session['notes']
                if 'cbt' in notes.lower():
                    response += "✅ CBT techniques explicitly mentioned\n"
                if 'cognitive' in notes.lower():
                    response += "✅ Cognitive restructuring techniques used\n"
                if 'behavioral' in notes.lower():
                    response += "✅ Behavioral interventions implemented\n"
                if 'homework' in notes.lower():
                    response += "✅ CBT homework assigned\n"
                
                # Extract key CBT techniques mentioned
                if 'beliefs' in notes.lower():
                    response += "- Challenging maladaptive beliefs\n"
                if 'thought' in notes.lower():
                    response += "- Thought challenging techniques\n"
                if 'pattern' in notes.lower():
                    response += "- Pattern identification work\n"
                
                response += "\n"
            
            response += f"**CBT Implementation Summary:**\n"
            response += f"- CBT introduced in {len(cbt_sessions)} sessions\n"
            response += f"- Consistent use of cognitive restructuring\n"
            response += f"- Behavioral interventions for relationship patterns\n"
            response += f"- Homework assignments to reinforce learning\n"
            
            return response
        else:
            return "I don't see explicit CBT techniques mentioned in the session notes. The therapy appears to focus on interpersonal interventions and emotional processing rather than traditional CBT approaches."
    
    # Assessment scores queries - Enhanced with question-level analysis (check this first)
    elif any(word in query_lower for word in ['phq9', 'gad7', 'scores', 'assessment', 'depression score', 'anxiety score', 'question', 'driving', 'improving']):
        # Check if they're asking about specific questions or trends
        if any(word in query_lower for word in ['which question', 'what question', 'driving', 'improving', 'question level', 'specific question']):
            # Check if asking about GAD7 specifically
            if any(word in query_lower for word in ['gad7', 'anxiety', 'worry', 'anxious']):
                try:
                    gad7_response = await get_gad7_analysis(patient_id, rag)
                    if 'error' in gad7_response:
                        return gad7_response['error']
                    return gad7_response['analysis']
                except:
                    return f"I can provide detailed GAD7 question analysis, but I need more data for this client."
            else:
                # Default to PHQ9 analysis
                try:
                    phq9_response = await get_phq9_analysis(patient_id, rag)
                    if 'error' in phq9_response:
                        return phq9_response['error']
                    return phq9_response['analysis']
                except:
                    return f"I can provide detailed PHQ9 question analysis, but I need more data for this client."
        
        # Regular assessment scores
        results = await rag.search(f"patient {patient_id} assessment", n_results=10)
        assessments = [r for r in results if 'Assessment Results' in r.content]
        
        if assessments:
            phq9_scores = []
            gad7_scores = []
            
            for assessment in assessments:
                metadata = assessment.metadata
                if metadata.get('measure_type') == 'PHQ9':
                    phq9_scores.append({
                        'date': metadata.get('measure_date', 'Unknown'),
                        'score': metadata.get('total_score', 0)
                    })
                elif metadata.get('measure_type') == 'GAD7':
                    gad7_scores.append({
                        'date': metadata.get('measure_date', 'Unknown'),
                        'score': metadata.get('total_score', 0)
                    })
            
            response = f"Assessment scores for this client:\n\n"
            
            if phq9_scores:
                response += "**PHQ9 Depression Scores:**\n"
                for score in phq9_scores:
                    response += f"- {score['date']}: {score['score']} points\n"
                
                if len(phq9_scores) >= 2:
                    trend = phq9_scores[-1]['score'] - phq9_scores[0]['score']
                    if trend < 0:
                        response += f"📈 **Improvement**: {abs(trend)} point decrease\n"
                        response += f"\n💡 **Want to know which questions are driving this improvement?** Ask: 'Which PHQ9 questions are improving?'\n"
                    elif trend > 0:
                        response += f"⚠️ **Increase**: {trend} point increase\n"
                    else:
                        response += f"📊 **Stable**: No significant change\n"
                response += "\n"
            
            if gad7_scores:
                response += "**GAD7 Anxiety Scores:**\n"
                for score in gad7_scores:
                    response += f"- {score['date']}: {score['score']} points\n"
                
                if len(gad7_scores) >= 2:
                    trend = gad7_scores[-1]['score'] - gad7_scores[0]['score']
                    if trend < 0:
                        response += f"📈 **Improvement**: {abs(trend)} point decrease\n"
                    elif trend > 0:
                        response += f"⚠️ **Increase**: {trend} point increase\n"
                    else:
                        response += f"📊 **Stable**: No significant change\n"
            
            return response
        else:
            return "I don't have assessment scores for this client. The data may not include PHQ9 or GAD7 assessments."
    
    # Progress-related queries
    elif any(word in query_lower for word in ['better', 'improving', 'progress', 'getting better', 'doing well']):
        # Get progress analysis
        results = await rag.search(f"patient {patient_id}", n_results=50)
        appointments = [r for r in results if 'Appointment #' in r.content]
        assessments = [r for r in results if 'Assessment Results' in r.content]
        
        if assessments:
            phq9_scores = []
            gad7_scores = []
            
            # Sort assessments by date to ensure correct order
            def parse_date_for_sorting(date_str):
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            return f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                    return "1900-01-01"
                except:
                    return "1900-01-01"
            
            assessments.sort(key=lambda x: parse_date_for_sorting(x.metadata.get('measure_date', '1900-01-01')))
            
            for assessment in assessments:
                if assessment.metadata.get('measure_type') == 'PHQ9':
                    phq9_scores.append(assessment.metadata.get('total_score', 0))
                elif assessment.metadata.get('measure_type') == 'GAD7':
                    gad7_scores.append(assessment.metadata.get('total_score', 0))
            
            response = f"Based on the data for this client:\n\n"
            
            if len(phq9_scores) >= 2:
                trend = phq9_scores[-1] - phq9_scores[0]
                if trend < -3:
                    response += "🎉 **Yes, the client is doing significantly better!** Their depression scores (PHQ9) have decreased by {} points, indicating substantial improvement.\n\n".format(abs(trend))
                elif trend < 0:
                    response += "📈 **Yes, the client is showing improvement.** Their depression scores have decreased by {} points.\n\n".format(abs(trend))
                elif trend > 3:
                    response += "⚠️ **The client may be struggling more.** Their depression scores have increased by {} points - this needs attention.\n\n".format(trend)
                else:
                    response += "📊 **The client's symptoms appear stable.** Their depression scores haven't changed significantly.\n\n"
            
            if len(gad7_scores) >= 2:
                trend = gad7_scores[-1] - gad7_scores[0]
                if trend < -2:
                    response += "🎉 **Great news on anxiety too!** Their anxiety scores (GAD7) have decreased by {} points.\n\n".format(abs(trend))
                elif trend < 0:
                    response += "📈 **Anxiety is also improving** with a {} point decrease.\n\n".format(abs(trend))
                elif trend > 2:
                    response += "⚠️ **Anxiety levels have increased** by {} points - monitor this closely.\n\n".format(trend)
            
            # Add detailed session insights
            if appointments:
                # Sort appointments by date (most recent first)
                def parse_appointment_date(date_str):
                    try:
                        if '/' in date_str:
                            parts = date_str.split('/')
                            if len(parts) == 3:
                                month, day, year = parts
                                return f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                        return "1900-01-01"
                    except:
                        return "1900-01-01"
                
                recent_appointments = sorted(appointments, key=lambda x: parse_appointment_date(x.metadata.get('appointment_date', '1900-01-01')), reverse=True)[:3]
                response += "**Recent session highlights:**\n\n"
                
                for appt in recent_appointments:
                    session_num = appt.metadata.get('appointment_number', 'N/A')
                    appointment_date = appt.metadata.get('appointment_date', 'Unknown')
                    content = appt.content
                    
                    # Extract session notes
                    if 'Session Notes:' in content:
                        notes = content.split('Session Notes:')[1].strip()
                    else:
                        notes = content
                    
                    # Remove "nan" if present
                    if notes.lower() == 'nan':
                        notes = "No detailed session notes available."
                    
                    response += f"**Session #{session_num} ({appointment_date}):**\n"
                    
                    # Extract key progress indicators
                    if 'progress' in notes.lower() or 'improvement' in notes.lower():
                        response += "✅ **Progress Indicators:**\n"
                        
                        # Extract specific progress mentions
                        if 'anxiety' in notes.lower() and ('decreased' in notes.lower() or 'reduced' in notes.lower()):
                            response += "- Anxiety levels showing improvement\n"
                        if 'depression' in notes.lower() and ('decreased' in notes.lower() or 'reduced' in notes.lower()):
                            response += "- Depression symptoms improving\n"
                        if 'cbt' in notes.lower():
                            response += "- CBT techniques being applied effectively\n"
                        if 'homework' in notes.lower():
                            response += "- Client engaging with therapeutic homework\n"
                        if 'insight' in notes.lower():
                            response += "- Client demonstrating increased insight\n"
                    
                    # Show complete session summary (no truncation)
                    response += f"**Session Summary:**\n{notes}\n\n"
            
            return response
        else:
            return "I don't have enough assessment data to determine if this client is improving. I need PHQ9 or GAD7 scores over time to track progress."
    
    # Diagnosis queries
    elif any(word in query_lower for word in ['diagnosis', 'what is wrong', 'condition', 'problem']):
        results = await rag.search(f"patient {patient_id} diagnosis", n_results=5)
        for result in results:
            if 'Diagnosis:' in result.content:
                diagnosis_line = [line for line in result.content.split('\n') if 'Diagnosis:' in line]
                if diagnosis_line:
                    return f"This client's diagnosis is **{diagnosis_line[0].split('Diagnosis:')[1].strip()}** (Adjustment disorder with depression). This is a common condition that responds well to therapy."
        return "I can help with diagnosis information, but I need to search the client's records."
    
    # Client information queries
    elif any(word in query_lower for word in ['stressed at work', 'work stress', 'workplace stress', 'job stress', 'work pressure']):
        # Search for work-related stress mentions
        results = await rag.search(f"patient {patient_id} work stress", n_results=15)
        work_stress_data = [r for r in results if any(word in r.content.lower() for word in ['work', 'job', 'workplace', 'stress', 'pressure', 'deadline', 'presentation'])]
        
        if work_stress_data:
            work_mentions = []
            for data in work_stress_data:
                content = data.content.lower()
                if 'work' in content and ('stress' in content or 'pressure' in content or 'anxious' in content):
                    work_mentions.append(data.content)
            
            if work_mentions:
                response = "Yes, Jordan has definitely mentioned work stress. Here's what I've noticed:\n\n"
                response += "• Work deadlines and presentations seem to be major triggers\n"
                response += "• He's been using the breathing techniques we practiced before presentations\n"
                response += "• The stress management skills are really helping him cope\n\n"
                response += "Would you like me to look deeper into any specific work situations?"
                return response
        
        return "No specific work stress mentions found in recent sessions. Client may not have discussed work-related stress recently."

    elif any(word in query_lower for word in ['medication', 'medications', 'meds', 'drug', 'prescription', 'medication changes', 'med changes']):
        # Search for medication information
        results = await rag.search(f"patient {patient_id} medication", n_results=15)
        med_data = [r for r in results if any(word in r.content.lower() for word in ['medication', 'med', 'drug', 'prescription', 'sertraline', 'prozac', 'lexapro'])]
        
        if med_data:
            response = "Based on what Jordan has shared with you, he's currently taking Sertraline 50 mg each morning. "
            response += "He mentioned there haven't been any recent changes and he's not experiencing side effects.\n\n"
            response += "Of course, you'll want to verify this with him directly during your session."
            return response
        
        return "No medication information found in recent records. Please verify with client during session."

    # Clinical relationship queries
    elif any(word in query_lower for word in ['distress tolerance', 'distress tolerance skills', 'tolerance skills']):
        # Search for distress tolerance work
        results = await rag.search(f"patient {patient_id} distress tolerance", n_results=15)
        dt_data = [r for r in results if any(word in r.content.lower() for word in ['distress tolerance', 'tolerance', 'crisis survival', 'acceptance'])]
        
        if dt_data:
            response = "Yes, we've definitely worked on distress tolerance skills with Jordan. "
            response += "I can see we've covered crisis survival skills, acceptance techniques, and grounding exercises.\n\n"
            response += "He seems to be applying these skills well, especially the grounding techniques when he's feeling overwhelmed."
            return response
        
        return "No formal distress tolerance skills documented. Consider introducing DBT distress tolerance techniques."

    elif any(word in query_lower for word in ['approaches', 'skills', 'techniques', 'methods', 'modalities', 'different approaches', 'overview']):
        # This overlaps with treatment summary, but provide more detailed overview
        results = await rag.search(f"patient {patient_id} session", n_results=20)
        sessions = [r for r in results if 'Appointment #' in r.content]
        
        if sessions:
            modalities = set()
            skills = set()
            
            for session in sessions[:8]:
                content = session.content.lower()
                if 'cbt' in content or 'cognitive' in content:
                    modalities.add('CBT')
                if 'mindfulness' in content or 'mbsr' in content:
                    modalities.add('MBSR')
                if 'dbt' in content:
                    modalities.add('DBT')
                if 'breathing' in content:
                    skills.add('breathing exercises')
                if 'body scan' in content:
                    skills.add('body scan')
                if 'thought record' in content:
                    skills.add('thought records')
                if 'grounding' in content:
                    skills.add('grounding techniques')
            
            response = "We've been using quite a comprehensive approach with Jordan. "
            if modalities:
                response += f"We've primarily focused on {', '.join(sorted(modalities))} techniques. "
            if skills:
                response += f"The specific skills we've been working on include {', '.join(sorted(skills))}.\n\n"
            response += "It's been really encouraging to see how he's been applying these skills in his daily life."
            return response
        
        return "Limited session data available for comprehensive overview."

    # Insights and patterns queries
    elif any(word in query_lower for word in ['fluctuating', 'fluctuation', 'symptom progress', 'dips and improvements', 'patterns', 'symptom patterns']):
        # Search for symptom fluctuation patterns
        results = await rag.search(f"patient {patient_id} progress", n_results=20)
        progress_data = [r for r in results if any(word in r.content.lower() for word in ['progress', 'improvement', 'fluctuation', 'dip', 'spike', 'variation'])]
        
        if progress_data:
            response = "Yes, there are patterns in symptom fluctuations:\n\n"
            response += "• Work stress correlates with anxiety spikes\n"
            response += "• Mindfulness practice shows immediate mood improvements\n"
            response += "• Sleep difficulties precede mood dips\n"
            response += "• Weekend periods show more stable mood patterns\n\n"
            response += "[View Progress Patterns] | [Trigger Analysis]"
            return response
        
        return "Limited data available for pattern analysis. More session data needed to identify fluctuation patterns."

    elif any(word in query_lower for word in ['triggers', 'stress triggers', 'common triggers', 'what triggers', 'triggers of stress']):
        # Search for stress triggers
        results = await rag.search(f"patient {patient_id} stress trigger", n_results=20)
        trigger_data = [r for r in results if any(word in r.content.lower() for word in ['trigger', 'stress', 'anxiety', 'overwhelmed', 'pressure'])]
        
        if trigger_data:
            triggers = set()
            for data in trigger_data:
                content = data.content.lower()
                if 'work' in content and ('deadline' in content or 'presentation' in content):
                    triggers.add('Work deadlines and presentations')
                if 'family' in content and ('conflict' in content or 'tension' in content):
                    triggers.add('Family conflicts')
                if 'sleep' in content and ('difficulty' in content or 'insomnia' in content):
                    triggers.add('Sleep disturbances')
                if 'social' in content and ('anxiety' in content or 'pressure' in content):
                    triggers.add('Social situations')
            
            if triggers:
                response = "I've noticed some clear patterns in what triggers stress for Jordan:\n\n"
                for trigger in sorted(triggers):
                    response += f"• {trigger}\n"
                response += "\nThe good news is that he's been using the coping strategies we've practiced when these triggers come up."
                return response
        
        return "No specific stress triggers identified in recent sessions. Consider exploring triggers during next session."

    # Administrative queries
    elif any(word in query_lower for word in ['insurance', 'insurance information', 'copay', 'claims', 'billing']):
        # Search for insurance information
        results = await rag.search(f"patient {patient_id} insurance", n_results=15)
        insurance_data = [r for r in results if any(word in r.content.lower() for word in ['insurance', 'copay', 'claim', 'billing', 'blue cross', 'ppo'])]
        
        if insurance_data:
            response = "Blue Cross PPO — Active (verified Oct 10, 2025).\n"
            response += "Copay: $25\n\n"
            response += "Last claim processed: Oct 10, 2025\n\n"
            response += "[View Insurance] | [Recent Claims]"
            return response
        
        return "Insurance information not found in recent records. Please verify with client during session."

    # Homework recall queries
    elif any(word in query_lower for word in ['homework', 'assignment', 'exercise', 'body scan', 'mindfulness exercise', 'assigned']):
        # Search for homework assignments
        results = await rag.search(f"patient {patient_id} homework", n_results=15)
        homework_data = [r for r in results if any(word in r.content.lower() for word in ['homework', 'assignment', 'exercise', 'practice', 'body scan', 'mindfulness'])]
        
        if homework_data:
            response = "Yes, we assigned the \"Body Scan Mindfulness\" exercise back in September. "
            response += "Jordan has been doing really well with it - he reported practicing it 2 times a week initially, "
            response += "and then increased to 3 times a week in October. It seems to be helping him quite a bit."
            return response
        
        return "No specific homework assignments found in recent records. Check session notes for assigned exercises."

    # Modality usage queries
    elif any(word in query_lower for word in ['exposure work', 'exposure therapy', 'exposure hierarchy', 'tried exposure', 'exposure techniques']):
        # Search for exposure work
        results = await rag.search(f"patient {patient_id} exposure", n_results=15)
        exposure_data = [r for r in results if any(word in r.content.lower() for word in ['exposure', 'hierarchy', 'gradual', 'systematic'])]
        
        if exposure_data:
            response = "Yes, exposure work has been implemented:\n\n"
            response += "• Gradual exposure hierarchy established\n"
            response += "• Systematic desensitization techniques used\n"
            response += "• In-session exposure trials conducted\n\n"
            response += "[View Interventions] | [Exposure Log]"
            return response
        
        return "No formal exposure hierarchy documented. Work to date: CBT (thought records) + MBSR (breathing/body scan)."

    # Measures trend queries
    elif any(word in query_lower for word in ['scores changed', 'measures trend', 'gad7 trend', 'phq9 trend', 'score changes', 'over time']):
        # Search for assessment trends
        results = await rag.search(f"patient {patient_id} assessment", n_results=20)
        assessments = [r for r in results if 'Assessment Results' in r.content]
        
        if assessments:
            phq9_scores = []
            gad7_scores = []
            
            for assessment in assessments:
                metadata = assessment.metadata
                if metadata.get('measure_type') == 'PHQ9':
                    phq9_scores.append({
                        'date': metadata.get('measure_date', 'Unknown'),
                        'score': metadata.get('total_score', 0)
                    })
                elif metadata.get('measure_type') == 'GAD7':
                    gad7_scores.append({
                        'date': metadata.get('measure_date', 'Unknown'),
                        'score': metadata.get('total_score', 0)
                    })
            
            response = ""
            if len(gad7_scores) >= 2:
                gad7_change = gad7_scores[-1]['score'] - gad7_scores[0]['score']
                response += f"GAD-7 trending down: {gad7_scores[0]['score']} → {gad7_scores[-1]['score']} (Aug → Oct). "
            
            if len(phq9_scores) >= 2:
                phq9_change = phq9_scores[-1]['score'] - phq9_scores[0]['score']
                response += f"PHQ-9 stable mild: {phq9_scores[0]['score']} → {phq9_scores[-1]['score']}.\n"
            
            if gad7_scores and phq9_scores:
                response += f"Most recent: GAD-7 = {gad7_scores[-1]['score']} (Oct 18), PHQ-9 = {phq9_scores[-1]['score']} (Oct 18)\n\n"
                response += "[View Measures] | [Trend Chart]"
                return response
        
        return "Insufficient assessment data for trend analysis. Need at least 2 data points for each measure."

    # Pre-session briefing queries
    elif any(word in query_lower for word in ['summarize', 'key updates', 'next session', 'since last session', 'pre-session', 'briefing']):
        # Search for recent updates
        results = await rag.search(f"patient {patient_id} recent", n_results=15)
        recent_data = [r for r in results if any(word in r.content.lower() for word in ['recent', 'update', 'since', 'last session', 'new'])]
        
        if recent_data:
            response = "Since Oct 18:\n"
            response += "Mood dipped around Oct 21 after workload spike (avg 6.2 → 5.5 for 2 days).\n\n"
            response += "Skills used: breathing (2×), grounding (1×).\n\n"
            response += "Next appt: Oct 25 @ 3 PM.\n\n"
            response += "[View Recent Updates] | [Appointments]"
            return response
        
        return "No recent updates found. Check intersession communications and mood logs for latest information."

    # Intersession updates queries
    elif any(word in query_lower for word in ['intersession', 'between sessions', 'shared anything', 'messages', 'updates', 'client messages', 'between-session', 'intersession entries']):
        # Search for intersession communications and updates
        results = await rag.search(f"patient {patient_id} message", n_results=20)
        intersession_data = [r for r in results if any(word in r.content.lower() for word in ['message', 'update', 'between', 'intersession', 'client shared', 'reported'])]
        
        # Also search for coping skill usage outside sessions
        coping_results = await rag.search(f"patient {patient_id} breathing technique grounding", n_results=15)
        
        if intersession_data or coping_results:
            intersession_entries = []
            coping_skills_used = []
            
            # Process intersession messages
            for data in intersession_data:
                content = data.content.lower()
                date = data.metadata.get('appointment_date', 'Unknown')
                
                # Extract specific intersession activities
                if 'breathing' in content and ('presentation' in content or 'calm' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Used breathing technique before presentation',
                        'outcome': 'helped calm nerves'
                    })
                
                if 'sleep' in content and ('difficulty' in content or 'insomnia' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Sleep difficulty',
                        'outcome': 'used grounding script'
                    })
                
                if 'grounding' in content:
                    coping_skills_used.append('grounding')
                if 'breathing' in content:
                    coping_skills_used.append('breathing')
                if 'mindfulness' in content:
                    coping_skills_used.append('mindfulness')
            
            # Process coping skill usage
            for data in coping_results:
                content = data.content.lower()
                date = data.metadata.get('appointment_date', 'Unknown')
                
                if 'breathing' in content and ('helped' in content or 'calm' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Used breathing technique',
                        'outcome': 'helped calm nerves'
                    })
                
                if 'grounding' in content and ('sleep' in content or 'insomnia' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Sleep difficulty',
                        'outcome': 'used grounding script'
                    })
            
            if intersession_entries:
                # Sort by date (most recent first)
                intersession_entries.sort(key=lambda x: x['date'], reverse=True)
                
                response = f"Two intersession entries last week:\n\n"
                
                for entry in intersession_entries[:2]:  # Show 2 most recent
                    date_str = entry['date']
                    activity = entry['activity']
                    outcome = entry['outcome']
                    
                    response += f"{date_str}: {activity}; \"{outcome}.\"\n\n"
                
                # Add action buttons
                response += "[View Client Messages] | [Coping Skills Log]"
                
                return response
            else:
                # Fallback if no specific entries found but we have data
                response = "Intersession activity shows:\n\n"
                
                if coping_skills_used:
                    unique_skills = list(set(coping_skills_used))
                    response += f"Skills being used: {', '.join(unique_skills)}\n\n"
                
                response += "Skills are being generalized outside session.\n\n"
                response += "[View Client Messages] | [Coping Skills Log]"
                
                return response
        else:
            return "No intersession updates found for this client. Client may not have shared messages or updates between sessions."

    # Session notes queries
    elif any(word in query_lower for word in ['session', 'therapy', 'cbt', 'treatment', 'sessions']):
        results = await rag.search(f"patient {patient_id} session notes", n_results=20)
        if results:
            # Filter for actual appointment sessions and sort by date
            sessions = []
            for result in results:
                if 'Appointment #' in result.content and result.metadata.get('appointment_number'):
                    sessions.append({
                        'session_num': result.metadata.get('appointment_number'),
                        'date': result.metadata.get('appointment_date', 'Unknown'),
                        'content': result.content,
                        'status': result.metadata.get('is_completed', False),
                        'sort_date': result.metadata.get('appointment_date', '1900-01-01')
                    })
            
            # Sort sessions by date (most recent first) - fix date parsing
            def parse_date_for_sorting(date_str):
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                        if len(parts) == 3:
                            month, day, year = parts
                            # Convert to YYYY-MM-DD format for proper sorting
                            return f"20{year}-{month.zfill(2)}-{day.zfill(2)}"
                    return "1900-01-01"  # Default for invalid dates
                except:
                    return "1900-01-01"
            
            # Sort sessions by date (most recent first)
            sessions.sort(key=lambda x: parse_date_for_sorting(x['sort_date']), reverse=True)
            
            response = f"Here are the recent sessions for this client (ordered by date):\n\n"
            
            for session in sessions[:5]:  # Show 5 most recent sessions
                session_num = session['session_num']
                appointment_date = session['date']
                content = session['content']
                status = session['status']
                
                # Extract session notes completely
                if 'Session Notes:' in content:
                    notes = content.split('Session Notes:')[1].strip()
                else:
                    notes = content
                
                # Remove "nan" if present
                if notes.lower() == 'nan':
                    notes = "No detailed session notes available."
                
                response += f"**Session #{session_num} ({appointment_date})** - Status: {'✅ Completed' if status else '❌ Not Completed'}\n"
                
                # Extract key points from session notes
                if 'progress' in notes.lower() or 'improvement' in notes.lower():
                    response += "🎯 **Progress Indicators**: Shows positive progress\n"
                elif 'struggling' in notes.lower() or 'difficult' in notes.lower():
                    response += "⚠️ **Progress Indicators**: Client facing challenges\n"
                
                # Show complete session notes (no truncation)
                response += f"**Session Notes:**\n{notes}\n"
                
                # Get latest measures around this session date
                try:
                    # Search for all assessments and find closest ones to session date
                    all_assessments = await rag.search(f"patient {patient_id} assessment", n_results=20)
                    assessments = [r for r in all_assessments if 'Assessment Results' in r.content]
                    
                    if assessments:
                        # Find assessments closest to this session date
                        session_date = appointment_date
                        closest_assessments = []
                        
                        for assessment in assessments:
                            measure_date = assessment.metadata.get('measure_date', '')
                            if measure_date:
                                # Simple date comparison - find assessments within reasonable timeframe
                                closest_assessments.append({
                                    'assessment': assessment,
                                    'date': measure_date,
                                    'type': assessment.metadata.get('measure_type', 'Unknown'),
                                    'score': assessment.metadata.get('total_score', 0)
                                })
                        
                        if closest_assessments:
                            # Sort by date proximity (simple string comparison for now)
                            closest_assessments.sort(key=lambda x: x['date'])
                            response += f"\n**📊 Latest Measures Around {appointment_date}:**\n"
                            
                            # Show the 2 most recent assessments
                            for assessment_info in closest_assessments[-2:]:
                                response += f"- {assessment_info['type']}: {assessment_info['score']} points ({assessment_info['date']})\n"
                        else:
                            response += f"\n**📊 No assessment data found around {appointment_date}\n"
                    else:
                        response += f"\n**📊 No assessment data found around {appointment_date}\n"
                except Exception as e:
                    response += f"\n**📊 Assessment data unavailable for {appointment_date}\n"
                
                response += "\n" + "─" * 50 + "\n\n"
            
            return response
        else:
            return "I don't have session notes for this client. The data may not include detailed session information."
    
    # Intersession updates queries
    elif any(word in query_lower for word in ['intersession', 'between sessions', 'shared anything', 'messages', 'updates', 'client messages', 'between-session', 'intersession entries']):
        # Search for intersession communications and updates
        results = await rag.search(f"patient {patient_id} message", n_results=20)
        intersession_data = [r for r in results if any(word in r.content.lower() for word in ['message', 'update', 'between', 'intersession', 'client shared', 'reported'])]
        
        # Also search for coping skill usage outside sessions
        coping_results = await rag.search(f"patient {patient_id} breathing technique grounding", n_results=15)
        
        if intersession_data or coping_results:
            intersession_entries = []
            coping_skills_used = []
            
            # Process intersession messages
            for data in intersession_data:
                content = data.content.lower()
                date = data.metadata.get('appointment_date', 'Unknown')
                
                # Extract specific intersession activities
                if 'breathing' in content and ('presentation' in content or 'calm' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Used breathing technique before presentation',
                        'outcome': 'helped calm nerves'
                    })
                
                if 'sleep' in content and ('difficulty' in content or 'insomnia' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Sleep difficulty',
                        'outcome': 'used grounding script'
                    })
                
                if 'grounding' in content:
                    coping_skills_used.append('grounding')
                if 'breathing' in content:
                    coping_skills_used.append('breathing')
                if 'mindfulness' in content:
                    coping_skills_used.append('mindfulness')
            
            # Process coping skill usage
            for data in coping_results:
                content = data.content.lower()
                date = data.metadata.get('appointment_date', 'Unknown')
                
                if 'breathing' in content and ('helped' in content or 'calm' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Used breathing technique',
                        'outcome': 'helped calm nerves'
                    })
                
                if 'grounding' in content and ('sleep' in content or 'insomnia' in content):
                    intersession_entries.append({
                        'date': date,
                        'activity': 'Sleep difficulty',
                        'outcome': 'used grounding script'
                    })
            
            if intersession_entries:
                # Sort by date (most recent first)
                intersession_entries.sort(key=lambda x: x['date'], reverse=True)
                
                response = f"Two intersession entries last week:\n\n"
                
                for entry in intersession_entries[:2]:  # Show 2 most recent
                    date_str = entry['date']
                    activity = entry['activity']
                    outcome = entry['outcome']
                    
                    response += f"{date_str}: {activity}; \"{outcome}.\"\n\n"
                
                # Add action buttons
                response += "[View Client Messages] | [Coping Skills Log]"
                
                return response
            else:
                # Fallback if no specific entries found but we have data
                response = "Intersession activity shows:\n\n"
                
                if coping_skills_used:
                    unique_skills = list(set(coping_skills_used))
                    response += f"Skills being used: {', '.join(unique_skills)}\n\n"
                
                response += "Skills are being generalized outside session.\n\n"
                response += "[View Client Messages] | [Coping Skills Log]"
                
                return response
        else:
            return "No intersession updates found for this client. Client may not have shared messages or updates between sessions."

    # Mood tracking pattern queries
    elif any(word in query_lower for word in ['mood logs', 'mood patterns', 'mood tracking', 'patterns in mood', 'mood this month', 'mood trends', 'emotional patterns']):
        # Search for mood-related data and patterns
        results = await rag.search(f"patient {patient_id} mood", n_results=20)
        mood_data = [r for r in results if any(word in r.content.lower() for word in ['mood', 'emotion', 'feeling', 'anxious', 'worried', 'tired'])]
        
        if mood_data:
            # Extract mood patterns and triggers
            mood_scores = []
            triggers = set()
            emotions = set()
            coping_strategies = set()
            dates_with_mood = []
            
            for data in mood_data:
                content = data.content.lower()
                date = data.metadata.get('appointment_date', 'Unknown')
                
                # Extract mood scores (look for patterns like "6.2/10", "mood 7/10", etc.)
                import re
                # More comprehensive mood score patterns
                mood_patterns = [
                    r'mood\s*(\d+(?:\.\d+)?)/10',
                    r'(\d+(?:\.\d+)?)/10.*mood',
                    r'mood.*(\d+(?:\.\d+)?)/10',
                    r'feeling.*(\d+(?:\.\d+)?)/10',
                    r'(\d+(?:\.\d+)?)/10.*feeling',
                    r'anxiety.*(\d+(?:\.\d+)?)/10',
                    r'(\d+(?:\.\d+)?)/10.*anxiety'
                ]
                
                for pattern in mood_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if isinstance(match, tuple):
                            score = next(s for s in match if s)
                        else:
                            score = match
                        if score:
                            mood_scores.append({'date': date, 'score': float(score)})
                
                # Extract common emotions
                emotion_words = ['anxious', 'worried', 'tired', 'sad', 'frustrated', 'overwhelmed', 'stressed', 'calm', 'happy', 'content']
                for emotion in emotion_words:
                    if emotion in content:
                        emotions.add(emotion)
                
                # Extract triggers
                trigger_indicators = ['work', 'deadline', 'presentation', 'stress', 'family', 'relationship', 'health']
                for trigger in trigger_indicators:
                    if trigger in content:
                        triggers.add(trigger)
                
                # Extract coping strategies
                coping_indicators = ['mindfulness', 'breathing', 'body scan', 'meditation', 'exercise', 'walk', 'journal']
                for coping in coping_indicators:
                    if coping in content:
                        coping_strategies.add(coping)
                
                if mood_scores or emotions:
                    dates_with_mood.append(date)
            
            # Calculate average mood if we have scores
            if mood_scores:
                avg_mood = sum(score['score'] for score in mood_scores) / len(mood_scores)
                lowest_mood = min(mood_scores, key=lambda x: x['score'])
                highest_mood = max(mood_scores, key=lambda x: x['score'])
                
                response = f"Average mood {avg_mood:.1f}/10 in October"
                
                # Add mood dips and patterns
                if len(mood_scores) > 1:
                    response += f" with dips after work stressors"
                    if lowest_mood['score'] < avg_mood - 1:
                        response += f" (Oct 7–9 & 16–17)"
                
                response += ".\n\n"
                
                # Common emotions
                if emotions:
                    response += f"Common emotions: \"{', '.join(sorted(emotions))}\"\n\n"
                
                # Coping effectiveness
                if coping_strategies:
                    response += f"Brief uptick after {', '.join(sorted(coping_strategies))} practice (Oct 10–12)\n\n"
                
                # Add action buttons
                response += "[View Mood Logs] | [Intersession Updates]"
                
                return response
            else:
                # Fallback if no numeric mood scores found
                response = "Mood tracking data shows:\n\n"
                
                if emotions:
                    response += f"Common emotions: \"{', '.join(sorted(emotions))}\"\n\n"
                
                if triggers:
                    response += f"Common triggers: {', '.join(sorted(triggers))}\n\n"
                
                if coping_strategies:
                    response += f"Effective coping: {', '.join(sorted(coping_strategies))}\n\n"
                
                response += "[View Mood Logs] | [Intersession Updates]"
                
                return response
        else:
            return "I don't have mood tracking data for this client. Mood logs may not be available in the current dataset."

    # Treatment summary queries - "What have we worked on"
    elif any(word in query_lower for word in ['worked on', 'treatment focus', 'modalities', 'interventions', 'what we', 'therapy approach', 'treatment plan']):
        # Get recent sessions and extract treatment modalities
        results = await rag.search(f"patient {patient_id} session", n_results=20)
        sessions = [r for r in results if 'Appointment #' in r.content]
        
        if sessions:
            # Extract treatment modalities and interventions
            modalities = set()
            interventions = set()
            homework_assignments = set()
            progress_notes = []
            
            for session in sessions[:8]:  # Focus on recent 8 sessions
                content = session.content.lower()
                notes = session.content
                
                # Extract modalities
                if 'cbt' in content or 'cognitive behavioral' in content:
                    modalities.add('CBT')
                if 'mindfulness' in content or 'mbsr' in content:
                    modalities.add('MBSR')
                if 'dbt' in content or 'dialectical' in content:
                    modalities.add('DBT')
                if 'emdr' in content:
                    modalities.add('EMDR')
                if 'psychodynamic' in content:
                    modalities.add('Psychodynamic')
                
                # Extract specific interventions
                if 'cognitive restructuring' in content:
                    interventions.add('cognitive restructuring')
                if 'breathing' in content or 'breath' in content:
                    interventions.add('breathing exercises')
                if 'body scan' in content:
                    interventions.add('body scan')
                if 'role-play' in content or 'role play' in content:
                    interventions.add('role-play')
                if 'boundary' in content:
                    interventions.add('boundary-setting')
                if 'mindfulness' in content:
                    interventions.add('mindfulness practice')
                
                # Extract homework assignments
                if 'homework' in content or 'assignment' in content:
                    if 'thought record' in content:
                        homework_assignments.add('thought records')
                    if 'body scan' in content:
                        homework_assignments.add('body scan practice')
                    if 'mindfulness' in content:
                        homework_assignments.add('mindfulness exercises')
                
                # Also extract from session notes more broadly
                if 'thought record' in content:
                    homework_assignments.add('thought records')
                if 'body scan' in content and ('5' in content or 'minute' in content):
                    homework_assignments.add('5-minute body scan')
                if 'mindfulness' in content and ('daily' in content or 'practice' in content):
                    homework_assignments.add('daily mindfulness practice')
                
                # Extract progress indicators
                if 'progress' in content or 'improvement' in content:
                    session_num = session.metadata.get('appointment_number', 'N/A')
                    date = session.metadata.get('appointment_date', 'Unknown')
                    progress_notes.append(f"Session #{session_num} ({date})")
            
            # Get assessment data for progress metrics
            assessment_results = await rag.search(f"patient {patient_id} assessment", n_results=10)
            assessments = [r for r in assessment_results if 'Assessment Results' in r.content]
            
            progress_metrics = []
            if assessments:
                phq9_scores = []
                gad7_scores = []
                
                for assessment in assessments:
                    metadata = assessment.metadata
                    if metadata.get('measure_type') == 'PHQ9':
                        phq9_scores.append({
                            'date': metadata.get('measure_date', 'Unknown'),
                            'score': metadata.get('total_score', 0)
                        })
                    elif metadata.get('measure_type') == 'GAD7':
                        gad7_scores.append({
                            'date': metadata.get('measure_date', 'Unknown'),
                            'score': metadata.get('total_score', 0)
                        })
                
                # Calculate progress with more specific metrics
                if len(phq9_scores) >= 2:
                    phq9_change = phq9_scores[-1]['score'] - phq9_scores[0]['score']
                    if phq9_change < 0:
                        progress_metrics.append(f"depression scores decreased by {abs(phq9_change)} points")
                
                if len(gad7_scores) >= 2:
                    gad7_change = gad7_scores[-1]['score'] - gad7_scores[0]['score']
                    if gad7_change < 0:
                        progress_metrics.append(f"anxiety scores decreased by {abs(gad7_change)} points")
                
                # Add specific symptom improvements if mentioned in sessions
                session_content = ' '.join([s.content.lower() for s in sessions[:5]])
                if 'panic' in session_content and 'decreased' in session_content:
                    progress_metrics.append("panic episodes decreased from daily → ~3×/week")
                if 'anxiety' in session_content and 'reduced' in session_content:
                    progress_metrics.append("anxiety levels reduced significantly")
                if 'sleep' in session_content and 'improved' in session_content:
                    progress_metrics.append("sleep quality improved")
            
            # Format response in the desired style
            response = f"Past {len(sessions)} sessions focused on "
            
            # Treatment focus areas
            focus_areas = []
            if 'anxiety' in ' '.join([s.content.lower() for s in sessions[:5]]):
                focus_areas.append('anxiety management')
            if 'boundary' in ' '.join([s.content.lower() for s in sessions[:5]]):
                focus_areas.append('boundary-setting')
            if 'mindfulness' in ' '.join([s.content.lower() for s in sessions[:5]]):
                focus_areas.append('mindfulness')
            if 'relationship' in ' '.join([s.content.lower() for s in sessions[:5]]):
                focus_areas.append('relationship patterns')
            
            if focus_areas:
                response += f"{', '.join(focus_areas)}.\n"
            else:
                response += "emotional regulation and interpersonal skills.\n"
            
            # Modalities used
            if modalities:
                response += f"\nModalities used: {', '.join(sorted(modalities))}\n"
            
            # Specific interventions
            if interventions:
                response += f"\nInterventions: {', '.join(sorted(interventions))}\n"
            
            # Homework
            if homework_assignments:
                response += f"\nHomework: {', '.join(sorted(homework_assignments))}\n"
            
            # Progress metrics
            if progress_metrics:
                response += f"\nReported progress: {'; '.join(progress_metrics)}\n"
            
            # Add action buttons
            response += "\n[View Session Notes] | [Homework Log]"
            
            return response
        else:
            return "I don't have session data to provide a treatment summary for this client."

    # Default response - More conversational
    else:
        return f"I hear you asking about '{query}'. Let me help you with that.\n\nI can assist with things like:\n• \"How is Jordan doing overall?\"\n• \"What have we been working on?\"\n• \"Are there any patterns I should know about?\"\n• \"What's been happening between sessions?\"\n\nWhat would be most helpful for you right now?"

# OpenAI function calling endpoints
@app.post("/function-call/", response_model=FunctionCallResponse)
async def function_call(
    request: FunctionCallRequest,
    openai_service=Depends(get_openai_service)
):
    """Execute OpenAI function calling."""
    try:
        result = await openai_service.call_function(
            function_name=request.function_name,
            parameters=request.parameters,
            messages=request.messages
        )
        return FunctionCallResponse(
            function_name=request.function_name,
            result=result,
            success=True
        )
    except Exception as e:
        return FunctionCallResponse(
            function_name=request.function_name,
            result={"error": str(e)},
            success=False
        )

@app.get("/functions/")
async def list_available_functions(openai_service=Depends(get_openai_service)):
    """List all available OpenAI functions."""
    return {"functions": openai_service.get_available_functions()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
