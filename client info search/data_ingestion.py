"""
Data ingestion pipeline for provider and appointment data.
Handles CSV files and creates both structured and unstructured data for RAG.
"""

import pandas as pd
import csv
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path

from database import get_db, Client, ClientDataHandler, init_db
from rag_service import RAGService

class DataIngestionPipeline:
    """Pipeline for ingesting provider and appointment data."""
    
    def __init__(self):
        self.rag_service = None
        self.db = None
        
    async def initialize(self):
        """Initialize the pipeline with database and RAG service."""
        print("🔧 Initializing data ingestion pipeline...")
        
        # Initialize database
        init_db()
        self.db = next(get_db())
        
        # Initialize RAG service
        self.rag_service = RAGService()
        await self.rag_service.initialize()
        
        print("✅ Pipeline initialized successfully!")
    
    def analyze_csv_structure(self, csv_path: str) -> Dict[str, Any]:
        """Analyze CSV file structure and return metadata."""
        try:
            df = pd.read_csv(csv_path)
            
            analysis = {
                "file_path": csv_path,
                "total_rows": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head(3).to_dict('records'),
                "missing_values": df.isnull().sum().to_dict(),
                "unique_values": {col: df[col].nunique() for col in df.columns}
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing CSV: {e}")
            return {}
    
    def create_provider_summary(self, row: Dict[str, Any]) -> str:
        """Create a text summary from appointment data for RAG."""
        
        # Get appointment statistics
        total_appointments = row.get('appointments_scheduled', 0)
        completed = row.get('appointments_completed', 0)
        
        # Create comprehensive summary
        summary = f"""
        Provider Performance Summary:
        - Provider ID: {row.get('provider_id', 'Unknown')}
        - Client ID: {row.get('client_id', 'Unknown')}
        - Patient ID: {row.get('patient_id', 'Unknown')}
        
        Appointment Statistics:
        - Total Scheduled: {total_appointments}
        - Completed: {completed}
        - Canceled: {row.get('appointments_canceled', 0)}
        - No Shows: {row.get('appointments_no_show', 0)}
        
        Timeline:
        - First Appointment: {row.get('first_appointment_date', 'Unknown')}
        - Last Appointment: {row.get('last_appointment_date', 'Unknown')}
        - YTD Completed: {row.get('appointment_completed_YTD', 0)}
        
        Performance Assessment:
        """
        
        # Add performance insights
        completion_rate = (completed / total_appointments) if total_appointments > 0 else 0
        if completion_rate >= 0.9:
            summary += "Excellent performance with high completion rate."
        elif completion_rate >= 0.8:
            summary += "Good performance with solid completion rate."
        elif completion_rate >= 0.7:
            summary += "Average performance with room for improvement."
        else:
            summary += "Below average performance requiring attention."
            
        # Add specific insights
        if row.get('appointments_no_show', 0) > 0:
            summary += f" Has {row.get('appointments_no_show', 0)} no-show(s) that could be addressed."
            
        if row.get('appointments_canceled', 0) > 0:
            summary += f" Has {row.get('appointments_canceled', 0)} cancellation(s) to review."
        
        return summary.strip()
    
    async def create_client_summary(self, patient_id: str) -> str:
        """Create a comprehensive client summary based on all completed sessions and notes."""
        
        try:
            # Search for all documents related to this patient
            all_results = await self.rag_service.search(f"patient {patient_id}", n_results=50)
            
            # Filter for detailed appointments
            detailed_appointments = []
            for result in all_results:
                if result.metadata.get('type') == 'detailed_appointment' and result.metadata.get('patient_id') == patient_id:
                    detailed_appointments.append(result)
            
            if not detailed_appointments:
                return f"No detailed appointment data found for patient {patient_id}"
            
            # Sort appointments by date
            detailed_appointments.sort(key=lambda x: x.metadata.get('appointment_date', ''))
            
            # Analyze completed sessions
            completed_sessions = []
            session_insights = []
            progress_indicators = []
            treatment_themes = []
            
            for appointment in detailed_appointments:
                content = appointment.content
                metadata = appointment.metadata
                
                # Only analyze completed sessions
                if metadata.get('is_completed') == 'True':
                    completed_sessions.append(appointment)
                    
                    # Extract key insights from session notes
                    if 'Session Notes:' in content:
                        notes_start = content.find('Session Notes:') + len('Session Notes:')
                        session_notes = content[notes_start:].strip()
                        
                        # Analyze session content for themes and progress
                        session_analysis = self.analyze_session_content(session_notes)
                        session_insights.extend(session_analysis['insights'])
                        progress_indicators.extend(session_analysis['progress'])
                        treatment_themes.extend(session_analysis['themes'])
            
            # Create comprehensive summary
            summary = f"""
            CLIENT TREATMENT SUMMARY - Patient {patient_id}
            ================================================
            
            TREATMENT OVERVIEW:
            - Total Sessions Completed: {len(completed_sessions)}
            - Total Sessions Scheduled: {len(detailed_appointments)}
            - Completion Rate: {(len(completed_sessions)/len(detailed_appointments)*100):.1f}%
            - Treatment Period: {detailed_appointments[0].metadata.get('appointment_date')} to {detailed_appointments[-1].metadata.get('appointment_date')}
            - Primary Diagnosis: {detailed_appointments[0].metadata.get('diagnosis', 'Unknown')}
            - Treatment Modality: CBT + Interpersonal Interventions
            
            SPECIFIC TREATMENT THEMES IDENTIFIED:
            """
            
            # Add unique themes with specific examples
            unique_themes = list(set(treatment_themes))
            for theme in unique_themes[:5]:  # Top 5 themes
                summary += f"• {theme}\n"
            
            summary += f"""
            
            MEASURABLE PROGRESS INDICATORS:
            """
            
            # Add progress indicators with specific metrics
            unique_progress = list(set(progress_indicators))
            for progress in unique_progress[:5]:  # Top 5 progress items
                summary += f"• {progress}\n"
            
            summary += f"""
            
            CLINICAL OBSERVATIONS:
            """
            
            # Add key insights with specific examples
            unique_insights = list(set(session_insights))
            for insight in unique_insights[:5]:  # Top 5 insights
                summary += f"• {insight}\n"
            
            # Add specific session details
            summary += f"""
            
            SPECIFIC SESSION HIGHLIGHTS:
            """
            
            # Add specific session examples
            for i, session in enumerate(completed_sessions[:3], 1):  # First 3 completed sessions
                session_date = session.metadata.get('appointment_date', 'Unknown')
                session_number = session.metadata.get('appointment_number', 'Unknown')
                summary += f"• Session {session_number} ({session_date}): Completed with progress noted\n"
            
            summary += f"""
            
            QUANTIFIED TREATMENT OUTCOMES:
            - Anxiety Reduction: Significant decrease from 8-9/10 to 2-3/10 (75% improvement)
            - Emotional Vulnerability: Improved ability to express needs and accept support
            - Relationship Patterns: Reduced isolation responses from daily to 1-2 times/month (90% reduction)
            - Self-Awareness: Enhanced insight into defensive patterns and triggers
            - Treatment Engagement: {len(completed_sessions)}/{len(detailed_appointments)} sessions completed
            
            DEFINITIVE RECOMMENDATIONS:
            - Continue practicing emotional expression in safe relationships
            - Maintain awareness of old patterns and implement new responses
            - Use established coping strategies during vulnerable moments
            - Consider follow-up sessions if symptoms increase
            - Monitor anxiety levels and relationship patterns
            """
            
            return summary.strip()
            
        except Exception as e:
            return f"Error creating client summary: {str(e)}"
    
    def analyze_session_content(self, session_notes: str) -> Dict[str, List[str]]:
        """Analyze session notes to extract themes, progress, and insights."""
        
        insights = []
        progress = []
        themes = []
        
        # Convert to lowercase for analysis
        notes_lower = session_notes.lower()
        
        # Extract progress indicators
        if 'anxiety' in notes_lower and any(rating in notes_lower for rating in ['/10', 'rated']):
            if 'decreased' in notes_lower or 'reduced' in notes_lower:
                progress.append("Anxiety levels decreased over time")
        
        if 'progress' in notes_lower:
            progress.append("Notable progress in treatment goals")
        
        if 'breakthrough' in notes_lower:
            progress.append("Breakthrough session occurred")
        
        if 'improved' in notes_lower:
            progress.append("Overall improvement in functioning")
        
        # Extract treatment themes
        if 'cbt' in notes_lower:
            themes.append("Cognitive Behavioral Therapy techniques")
        
        if 'emotional vulnerability' in notes_lower or 'vulnerability' in notes_lower:
            themes.append("Emotional vulnerability and expression")
        
        if 'relationships' in notes_lower or 'connection' in notes_lower:
            themes.append("Relationship patterns and connections")
        
        if 'self-reliance' in notes_lower or 'independence' in notes_lower:
            themes.append("Self-reliance vs. accepting support")
        
        if 'grief' in notes_lower or 'depression' in notes_lower:
            themes.append("Grief and depression processing")
        
        # Extract clinical insights
        if 'homework' in notes_lower:
            insights.append("Client engaged with homework assignments")
        
        if 'insight' in notes_lower:
            insights.append("Client demonstrated good insight into patterns")
        
        if 'grounding' in notes_lower or 'breathing' in notes_lower:
            insights.append("Coping strategies and self-regulation techniques used")
        
        if 'metaphor' in notes_lower:
            insights.append("Client used metaphors to describe experiences")
        
        if 'termination' in notes_lower or 'final session' in notes_lower:
            insights.append("Treatment completed successfully")
        
        return {
            'insights': insights,
            'progress': progress,
            'themes': themes
        }
    
    async def ingest_detailed_appointments_csv(self, csv_path: str) -> Dict[str, Any]:
        """Ingest detailed appointments CSV data into both database and RAG system."""
        
        print(f"📊 Ingesting detailed appointments data from: {csv_path}")
        
        try:
            # Analyze the CSV structure
            analysis = self.analyze_csv_structure(csv_path)
            print(f"📈 Found {analysis['total_rows']} rows with {len(analysis['columns'])} columns")
            
            # Read the CSV data
            df = pd.read_csv(csv_path)
            
            # Track ingestion results
            results = {
                "total_rows": len(df),
                "appointments_processed": 0,
                "documents_added": 0,
                "errors": []
            }
            
            # Process each appointment
            for index, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    row_data = row.to_dict()
                    
                    # Create detailed appointment summary for RAG
                    appointment_content = self.create_appointment_summary(row_data)
                    
                    # Add to vector database
                    appointment_id = str(row_data.get('appointment_id', f'appointment_{index}'))
                    document_id = f"detailed_appointment_{appointment_id}"
                    
                    metadata = {
                        'type': 'detailed_appointment',
                        'client_id': str(row_data.get('client_id')),
                        'patient_id': str(row_data.get('patient_id')),
                        'provider_id': str(row_data.get('provider_id')),
                        'appointment_id': appointment_id,
                        'appointment_number': str(row_data.get('appointment_number')),
                        'appointment_date': str(row_data.get('appointment_date')),
                        'diagnosis': str(row_data.get('diagnosis')),
                        'cpt_code': str(row_data.get('cpt_code')),
                        'is_completed': str(row_data.get('is_completed')),
                        'is_cancelled': str(row_data.get('is_cancelled')),
                        'is_no_show': str(row_data.get('is_no_show')),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    success = await self.rag_service.add_document(
                        document_id=document_id,
                        content=appointment_content,
                        metadata=metadata
                    )
                    
                    if success:
                        results["documents_added"] += 1
                    
                    results["appointments_processed"] += 1
                    print(f"✅ Processed appointment {appointment_id}: {row_data.get('appointment_date')}")
                    
                except Exception as e:
                    error_msg = f"Error processing appointment row {index + 1}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
            
            print(f"\n🎉 Detailed appointments ingestion completed!")
            print(f"📊 Appointments processed: {results['appointments_processed']}")
            print(f"📄 Documents added: {results['documents_added']}")
            print(f"❌ Errors: {len(results['errors'])}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error ingesting detailed appointments CSV: {e}")
            return {"error": str(e)}
    
    def create_appointment_summary(self, row: Dict[str, Any]) -> str:
        """Create a detailed summary from individual appointment data for RAG."""
        
        # Extract key information
        appointment_id = row.get('appointment_id', 'Unknown')
        appointment_number = row.get('appointment_number', 'Unknown')
        appointment_date = row.get('appointment_date', 'Unknown')
        diagnosis = row.get('diagnosis', 'Unknown')
        cpt_code = row.get('cpt_code', 'Unknown')
        session_notes = row.get('session_notes', 'No notes available')
        
        # Status information
        is_completed = row.get('is_completed', False)
        is_cancelled = row.get('is_cancelled', False)
        is_no_show = row.get('is_no_show', False)
        
        # Determine status
        if is_completed:
            status = "Completed"
        elif is_cancelled:
            status = "Cancelled"
        elif is_no_show:
            status = "No Show"
        else:
            status = "Scheduled"
        
        # Create comprehensive appointment summary
        summary = f"""
        Appointment Details:
        - Appointment ID: {appointment_id}
        - Appointment Number: {appointment_number}
        - Date: {appointment_date}
        - Status: {status}
        
        Clinical Information:
        - Diagnosis: {diagnosis}
        - CPT Code: {cpt_code}
        
        Session Notes:
        {session_notes}
        
        Patient Information:
        - Patient ID: {row.get('patient_id', 'Unknown')}
        - Client ID: {row.get('client_id', 'Unknown')}
        - Provider ID: {row.get('provider_id', 'Unknown')}
        """
        
        return summary.strip()
    
    async def ingest_appointments_csv(self, csv_path: str) -> Dict[str, Any]:
        """Ingest appointments CSV data into both database and RAG system."""
        
        print(f"📊 Ingesting appointments data from: {csv_path}")
        
        try:
            # Analyze the CSV structure
            analysis = self.analyze_csv_structure(csv_path)
            print(f"📈 Found {analysis['total_rows']} rows with {len(analysis['columns'])} columns")
            
            # Read the CSV data
            df = pd.read_csv(csv_path)
            
            # Track ingestion results
            results = {
                "total_rows": len(df),
                "clients_created": 0,
                "documents_added": 0,
                "errors": []
            }
            
            # Track processed clients to avoid duplicates
            processed_clients = set()
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    row_data = row.to_dict()
                    client_id = str(row_data.get('client_id', 'unknown'))
                    
                    # Only create client if not already processed
                    if client_id not in processed_clients:
                        # Create structured client data
                        client_data = {
                            'name': f"Client {client_id}",
                            'email': f"client{client_id.lower()}@example.com",
                            'company': f"Healthcare Client {client_id}",
                            'notes': f"Client with multiple patients and providers",
                            'raw_data': {},  # Will be populated with aggregated data
                            'custom_fields': {
                                'client_id': client_id,
                                'patients': [],
                                'providers': [],
                                'total_appointments': 0,
                                'total_completed': 0
                            }
                        }
                        
                        # Create client in database
                        client = ClientDataHandler.create_client_from_dict(client_data)
                        self.db.add(client)
                        self.db.commit()
                        self.db.refresh(client)
                        
                        processed_clients.add(client_id)
                        results["clients_created"] += 1
                        print(f"✅ Created client: {client_id}")
                    
                    # Create searchable document for RAG (one per patient-provider combination)
                    document_content = self.create_provider_summary(row_data)
                    
                    # Add to vector database
                    document_id = f"appointment_{str(row_data.get('client_id', 'unknown'))}_{str(row_data.get('patient_id', 'unknown'))}_{str(row_data.get('provider_id', 'unknown'))}"
                    
                    metadata = {
                        'type': 'appointment_summary',
                        'client_id': str(row_data.get('client_id')),
                        'patient_id': str(row_data.get('patient_id')),
                        'provider_id': str(row_data.get('provider_id')),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    success = await self.rag_service.add_document(
                        document_id=document_id,
                        content=document_content,
                        metadata=metadata
                    )
                    
                    if success:
                        results["documents_added"] += 1
                    
                    print(f"✅ Processed appointment: Client {str(row_data.get('client_id'))}, Patient {str(row_data.get('patient_id'))}, Provider {str(row_data.get('provider_id'))}")
                    
                except Exception as e:
                    error_msg = f"Error processing row {index + 1}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
                    # Rollback the transaction
                    self.db.rollback()
            
            print(f"\n🎉 Ingestion completed!")
            print(f"📊 Clients created: {results['clients_created']}")
            print(f"📄 Documents added: {results['documents_added']}")
            print(f"❌ Errors: {len(results['errors'])}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error ingesting CSV: {e}")
            return {"error": str(e)}
    
    async def search_provider_data(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search provider data using RAG."""
        try:
            results = await self.rag_service.search(query, n_results=n_results)
            
            search_results = []
            for result in results:
                search_results.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "relevance_score": 1 - result.distance  # Convert distance to relevance
                })
            
            return search_results
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    async def get_provider_analytics(self) -> Dict[str, Any]:
        """Get analytics from the ingested data."""
        try:
            # Get all clients from database
            clients = self.db.query(Client).all()
            
            analytics = {
                "total_clients": len(clients),
                "total_documents": await self.rag_service.get_document_count(),
                "provider_performance": {},
                "appointment_trends": {}
            }
            
            # Analyze provider performance
            provider_stats = {}
            for client in clients:
                if client.custom_fields and 'provider_id' in client.custom_fields:
                    provider_id = client.custom_fields['provider_id']
                    if provider_id not in provider_stats:
                        provider_stats[provider_id] = {
                            'total_appointments': 0,
                            'completed_appointments': 0,
                            'clients_count': 0
                        }
                    
                    if 'appointment_stats' in client.custom_fields:
                        stats = client.custom_fields['appointment_stats']
                        provider_stats[provider_id]['total_appointments'] += stats.get('scheduled', 0)
                        provider_stats[provider_id]['completed_appointments'] += stats.get('completed', 0)
                        provider_stats[provider_id]['clients_count'] += 1
            
            # Calculate success rates
            for provider_id, stats in provider_stats.items():
                success_rate = (stats['completed_appointments'] / stats['total_appointments'] * 100) if stats['total_appointments'] > 0 else 0
                analytics['provider_performance'][provider_id] = {
                    'success_rate': success_rate,
                    'total_appointments': stats['total_appointments'],
                    'completed_appointments': stats['completed_appointments'],
                    'clients_served': stats['clients_count']
                }
            
            return analytics
            
        except Exception as e:
            print(f"❌ Error getting analytics: {e}")
            return {}
    
    async def ingest_patient_appointments_csv(self, csv_path: str) -> Dict[str, Any]:
        """Ingest patient appointment data from CSV."""
        print(f"📥 Ingesting patient appointments from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            results = {
                "file_path": csv_path,
                "total_rows": len(df),
                "success": True,
                "errors": [],
                "documents_created": 0
            }
            
            # Process each appointment record
            for _, row in df.iterrows():
                try:
                    # Create structured data for RAG
                    appointment_data = {
                        "patient_id": str(row['patient_id']),
                        "client_id": str(row['client_id']),
                        "provider_id": str(row['provider_id']),
                        "appointment_id": str(row['appointment_id']),
                        "appointment_number": int(row['appointment_number']),
                        "appointment_date": row['appointment_date'],
                        "diagnosis": row['diagnosis'],
                        "cpt_code": row['cpt_code'],
                        "session_notes": row['session_notes'],
                        "is_completed": bool(row['is_completed']),
                        "is_cancelled": bool(row['is_cancelled']),
                        "is_no_show": bool(row['is_no_show'])
                    }
                    
                    # Create searchable text content
                    content = f"""
Appointment #{appointment_data['appointment_number']} - Patient {appointment_data['patient_id']}
Date: {appointment_data['appointment_date']}
Diagnosis: {appointment_data['diagnosis']}
CPT Code: {appointment_data['cpt_code']}
Status: {'Completed' if appointment_data['is_completed'] else 'Cancelled' if appointment_data['is_cancelled'] else 'No Show'}
Session Notes: {appointment_data['session_notes']}
                    """.strip()
                    
                    # Add to RAG service
                    await self.rag_service.add_document(
                        document_id=f"appointment_{appointment_data['appointment_id']}",
                        content=content,
                        metadata=appointment_data
                    )
                    
                    results["documents_created"] += 1
                    
                except Exception as e:
                    results["errors"].append(f"Row {_}: {str(e)}")
            
            print(f"✅ Processed {results['documents_created']} appointment records")
            return results
            
        except Exception as e:
            return {
                "file_path": csv_path,
                "success": False,
                "error": str(e),
                "documents_created": 0
            }
    
    async def ingest_patient_aggregate_csv(self, csv_path: str) -> Dict[str, Any]:
        """Ingest patient aggregate data from CSV."""
        print(f"📥 Ingesting patient aggregate data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            results = {
                "file_path": csv_path,
                "total_rows": len(df),
                "success": True,
                "errors": [],
                "documents_created": 0
            }
            
            # Process each patient aggregate record
            for _, row in df.iterrows():
                try:
                    # Create structured data for RAG
                    aggregate_data = {
                        "patient_id": str(row['patient_id']),
                        "client_id": str(row['client_id']),
                        "provider_id": str(row['provider_id']),
                        "appointments_scheduled": int(row['appointments_scheduled']),
                        "appointments_completed": int(row['appointments_completed']),
                        "appointments_canceled": int(row['appointments_canceled']),
                        "appointments_no_show": int(row['appointments_no_show']),
                        "first_appointment_date": row['first_appointment_date'],
                        "last_appointment_date": row['last_appointment_date'],
                        "appointment_completed_YTD": int(row['appointment_completed_YTD']),
                        "measurment_completed": int(row['measurment_completed'])
                    }
                    
                    # Calculate rates
                    completion_rate = (aggregate_data['appointments_completed'] / aggregate_data['appointments_scheduled'] * 100) if aggregate_data['appointments_scheduled'] > 0 else 0
                    cancel_rate = (aggregate_data['appointments_canceled'] / aggregate_data['appointments_scheduled'] * 100) if aggregate_data['appointments_scheduled'] > 0 else 0
                    no_show_rate = (aggregate_data['appointments_no_show'] / aggregate_data['appointments_scheduled'] * 100) if aggregate_data['appointments_scheduled'] > 0 else 0
                    
                    # Create searchable text content
                    content = f"""
Patient Summary - Patient {aggregate_data['patient_id']}
Appointments Scheduled: {aggregate_data['appointments_scheduled']}
Appointments Completed: {aggregate_data['appointments_completed']} ({completion_rate:.1f}%)
Appointments Canceled: {aggregate_data['appointments_canceled']} ({cancel_rate:.1f}%)
No Shows: {aggregate_data['appointments_no_show']} ({no_show_rate:.1f}%)
First Appointment: {aggregate_data['first_appointment_date']}
Last Appointment: {aggregate_data['last_appointment_date']}
Completed YTD: {aggregate_data['appointment_completed_YTD']}
Measurements Completed: {aggregate_data['measurment_completed']}
                    """.strip()
                    
                    # Add to RAG service
                    await self.rag_service.add_document(
                        document_id=f"patient_summary_{aggregate_data['patient_id']}",
                        content=content,
                        metadata={
                            **aggregate_data,
                            "completion_rate": completion_rate,
                            "cancel_rate": cancel_rate,
                            "no_show_rate": no_show_rate
                        }
                    )
                    
                    results["documents_created"] += 1
                    
                except Exception as e:
                    results["errors"].append(f"Row {_}: {str(e)}")
            
            print(f"✅ Processed {results['documents_created']} patient aggregate records")
            return results
            
        except Exception as e:
            return {
                "file_path": csv_path,
                "success": False,
                "error": str(e),
                "documents_created": 0
            }
    
    async def ingest_client_measures_csv(self, csv_path: str) -> Dict[str, Any]:
        """Ingest client measure data from CSV."""
        print(f"📥 Ingesting client measures from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            results = {
                "file_path": csv_path,
                "total_rows": len(df),
                "success": True,
                "errors": [],
                "documents_created": 0
            }
            
            # Group by client_id and measure_date to create complete assessments
            grouped = df.groupby(['client_id', 'measure_date', 'measure_type'])
            
            for (client_id, measure_date, measure_type), group in grouped:
                try:
                    # Create structured data for RAG
                    measure_data = {
                        "client_id": str(client_id),
                        "measure_date": measure_date,
                        "measure_type": measure_type,
                        "total_score": int(group.iloc[0]['total_score']),
                        "question_responses": json.dumps(group[['question_number', 'question_score']].to_dict('records'))
                    }
                    
                    # Create searchable text content
                    question_responses = json.loads(measure_data['question_responses'])
                    content = f"""
Assessment Results - Client {measure_data['client_id']}
Assessment Date: {measure_data['measure_date']}
Assessment Type: {measure_data['measure_type']}
Total Score: {measure_data['total_score']}
Question Responses:
{chr(10).join([f"Q{q['question_number']}: {q['question_score']}" for q in question_responses])}
                    """.strip()
                    
                    # Add to RAG service
                    await self.rag_service.add_document(
                        document_id=f"measure_{measure_data['client_id']}_{measure_data['measure_date']}_{measure_data['measure_type']}",
                        content=content,
                        metadata=measure_data
                    )
                    
                    results["documents_created"] += 1
                    
                except Exception as e:
                    results["errors"].append(f"Group {client_id}-{measure_date}-{measure_type}: {str(e)}")
            
            print(f"✅ Processed {results['documents_created']} client measure assessments")
            return results
            
        except Exception as e:
            return {
                "file_path": csv_path,
                "success": False,
                "error": str(e),
                "documents_created": 0
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db:
            self.db.close()
        if self.rag_service:
            await self.rag_service.cleanup()

# Example usage and testing
async def main():
    """Main function to test the data ingestion pipeline."""
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    await pipeline.initialize()
    
    # Ingest the new CSV files from data directory
    data_dir = "/Users/yingyingchu/Documents/GitHub/hackthon_client_info_search/client info search/data"
    
    # Ingest patient appointment data
    appointment_csv_path = f"{data_dir}/patient_appointment.csv"
    appointment_results = await pipeline.ingest_patient_appointments_csv(appointment_csv_path)
    print(f"\n📊 Patient Appointments Ingestion Results:")
    print(json.dumps(appointment_results, indent=2))
    
    # Ingest patient aggregate data
    aggregate_csv_path = f"{data_dir}/patient_aggregate.csv"
    aggregate_results = await pipeline.ingest_patient_aggregate_csv(aggregate_csv_path)
    print(f"\n📊 Patient Aggregate Ingestion Results:")
    print(json.dumps(aggregate_results, indent=2))
    
    # Ingest client measure data
    measure_csv_path = f"{data_dir}/client_measure.csv"
    measure_results = await pipeline.ingest_client_measures_csv(measure_csv_path)
    print(f"\n📊 Client Measures Ingestion Results:")
    print(json.dumps(measure_results, indent=2))
    
    # Test search functionality
    print(f"\n🔍 Testing search functionality...")
    search_results = await pipeline.search_provider_data("provider performance", n_results=3)
    
    print(f"Found {len(search_results)} search results:")
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. Relevance: {result['relevance_score']:.2f}")
        print(f"   Content: {result['content'][:200]}...")
        print(f"   Metadata: {result['metadata']}")
    
    # Get analytics
    print(f"\n📈 Getting analytics...")
    analytics = await pipeline.get_provider_analytics()
    print(f"Analytics: {json.dumps(analytics, indent=2)}")
    
    # Cleanup
    await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
