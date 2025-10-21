"""
OpenAI service with function calling capabilities.
"""

import openai
import json
import os
from typing import Dict, Any, List, Optional
import asyncio

class OpenAIService:
    """OpenAI service with function calling tools."""
    
    def __init__(self):
        self.client = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. OpenAI functionality will be limited.")
        else:
            openai.api_key = self.api_key
            self.client = openai
        
        # Define available functions
        self.functions = {
            "search_documents": {
                "name": "search_documents",
                "description": "Search for documents using RAG",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            "get_client_info": {
                "name": "get_client_info",
                "description": "Get information about a specific client",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "integer",
                            "description": "The client ID"
                        }
                    },
                    "required": ["client_id"]
                }
            },
            "create_client": {
                "name": "create_client",
                "description": "Create a new client",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Client name"
                        },
                        "email": {
                            "type": "string",
                            "description": "Client email"
                        },
                        "phone": {
                            "type": "string",
                            "description": "Client phone number"
                        },
                        "company": {
                            "type": "string",
                            "description": "Client company"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Additional notes about the client"
                        }
                    },
                    "required": ["name", "email"]
                }
            },
            "analyze_text": {
                "name": "analyze_text",
                "description": "Analyze and summarize text content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["summary", "sentiment", "keywords", "entities"],
                            "description": "Type of analysis to perform"
                        }
                    },
                    "required": ["text", "analysis_type"]
                }
            }
        }
    
    async def call_function(
        self, 
        function_name: str, 
        parameters: Dict[str, Any],
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Call a specific OpenAI function."""
        try:
            if not self.client:
                return {"error": "OpenAI client not initialized"}
            
            if function_name not in self.functions:
                return {"error": f"Function {function_name} not found"}
            
            # Prepare messages
            if not messages:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Execute function {function_name} with parameters: {parameters}"}
                ]
            
            # Add function definition
            function_def = self.functions[function_name]
            
            # Make API call
            response = await self.client.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                functions=[function_def],
                function_call={"name": function_name},
                temperature=0.1
            )
            
            # Extract function call result
            message = response.choices[0].message
            
            if message.function_call:
                function_name_called = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                return {
                    "function_name": function_name_called,
                    "arguments": function_args,
                    "response": message.content
                }
            else:
                return {
                    "response": message.content,
                    "function_called": False
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def chat_with_functions(
        self, 
        messages: List[Dict[str, str]],
        available_functions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Chat with OpenAI using available functions."""
        try:
            if not self.client:
                return {"error": "OpenAI client not initialized"}
            
            # Filter functions if specified
            functions_to_use = []
            if available_functions:
                functions_to_use = [
                    self.functions[func] for func in available_functions 
                    if func in self.functions
                ]
            else:
                functions_to_use = list(self.functions.values())
            
            # Make API call
            response = await self.client.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                functions=functions_to_use,
                function_call="auto",
                temperature=0.7
            )
            
            message = response.choices[0].message
            
            result = {
                "response": message.content,
                "function_call": None
            }
            
            if message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": json.loads(message.function_call.arguments)
                }
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        return list(self.functions.keys())
    
    def get_function_schema(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific function."""
        return self.functions.get(function_name)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            if not self.client:
                raise Exception("OpenAI client not initialized")
            
            response = await self.client.Embedding.acreate(
                model="text-embedding-ada-002",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            raise Exception(f"Error generating embedding: {e}")
    
    async def moderate_text(self, text: str) -> Dict[str, Any]:
        """Moderate text using OpenAI moderation API."""
        try:
            if not self.client:
                raise Exception("OpenAI client not initialized")
            
            response = await self.client.Moderation.acreate(input=text)
            
            return {
                "flagged": response.results[0].flagged,
                "categories": response.results[0].categories,
                "category_scores": response.results[0].category_scores
            }
            
        except Exception as e:
            raise Exception(f"Error moderating text: {e}")
