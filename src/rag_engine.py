import logging
import openai
import os
import json
import google.generativeai as genai
from typing import Dict, List, Any, Optional
from src.text_indexer import TextIndexer

class RAGEngine:
    """Core RAG engine for podcast search and topic-based querying."""
    
    def __init__(self):
        """Initialize the RAG engine."""
        # Load API keys
        self.openai_key = self._load_api_key("openai")
        self.gemini_key = self._load_api_key("gemini")
        
        # Set up OpenAI
        if self.openai_key:
            openai.api_key = self.openai_key
        
        # Set up Gemini
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
        
        # Load model preferences
        self.default_model = self._load_config().get("model", {}).get("default", "gemini")
        
        logging.info(f"RAG engine initialized with default model: {self.default_model}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json."""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
        return {}
    
    def _load_api_key(self, provider: str) -> Optional[str]:
        """Load API key from environment or config."""
        # Try environment variable first
        env_key = os.getenv(f"{provider.upper()}_API_KEY")
        if env_key:
            return env_key
        
        # Try loading from config file
        try:
            config = self._load_config()
            return config.get('api_keys', {}).get(provider)
        except Exception as e:
            logging.error(f"Error loading {provider} API key: {e}")
        
        return None
    
    def query_podcasts(self, query: str, text_indexer: TextIndexer, n_results: int = 5,
                      search_strategy: str = "semantic", model: str = None) -> Dict[str, Any]:
        """
        Query podcasts for specific topics with contextual understanding.
        
        Args:
            query: The search query
            text_indexer: TextIndexer instance
            n_results: Number of results to return
            search_strategy: Search strategy to use
            model: Model to use (gemini or openai)
            
        Returns:
            Dictionary with response and sources
        """
        try:
            # Use specified model or default
            if model is None:
                model = self.default_model
            
            # Search for relevant content
            search_results = text_indexer.search_similar_content(
                query=query,
                n_results=n_results,
                search_strategy=search_strategy
            )
            
            # Format context for LLM
            context = self.format_context(search_results)
            
            # Generate response using specified model
            if model == "gemini":
                response = self.generate_gemini_response(query, context, search_strategy)
            elif model == "openai":
                response = self.generate_openai_response(query, context, search_strategy)
            else:
                response = f"Error: Unknown model '{model}'. Available models: gemini, openai"
            
            return {
                "response": response,
                "sources": search_results,
                "search_info": {
                    "strategy": search_strategy,
                    "model": model,
                    "results_count": len(search_results.get('documents', [[]])[0])
                }
            }
            
        except Exception as e:
            logging.error(f"Error querying podcasts: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "sources": {"documents": [[]], "metadatas": [[]], "distances": [[]]},
                "search_info": {"strategy": search_strategy, "model": model, "results_count": 0}
            }
    
    def format_context(self, search_results: Dict[str, Any]) -> str:
        """Format search results into context for the LLM."""
        try:
            documents = search_results.get('documents', [[]])[0]
            metadatas = search_results.get('metadatas', [[]])[0]
            
            if not documents:
                return "No relevant content found."
            
            context_parts = []
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                if doc and metadata:
                    episode_title = metadata.get('episode_title', 'Unknown Episode')
                    start_time = metadata.get('start_time', 0)
                    end_time = metadata.get('end_time', 0)
                    speaker = metadata.get('speaker', 'Unknown Speaker')
                    
                    # Format timestamp
                    start_str = self._format_timestamp(start_time)
                    end_str = self._format_timestamp(end_time)
                    
                    context_part = f"""
Source {i+1} - {episode_title} ({start_str} - {end_str})
Speaker: {speaker}
Content: {doc}
"""
                    context_parts.append(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"Error formatting context: {e}")
            return "Error formatting context."
    
    def generate_gemini_response(self, query: str, context: str, search_strategy: str) -> str:
        """Generate response using Google Gemini."""
        try:
            if not self.gemini_key:
                return "Error: Gemini API key not configured. Please set your Gemini API key in config.json or environment variables."
            
            # Log the API key being used (first 10 chars for debugging)
            api_key_preview = self.gemini_key[:10] + "..." if self.gemini_key else "None"
            logging.info(f"Using Gemini API key: {api_key_preview}")
            
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""You are an AI assistant that helps users find information from podcast transcripts. 
Based on the provided context from podcast episodes, answer the user's question accurately and provide specific timestamps when possible.

Context from podcast transcripts:
{context}

User Question: {query}

Instructions:
1. Answer based only on the provided context
2. Include specific episode titles and timestamps in your response
3. If the context doesn't contain enough information, say so
4. Provide direct quotes when relevant
5. Format timestamps as MM:SS
6. Mention speakers when relevant
7. If searching across multiple episodes, highlight patterns or differences

Answer:"""

            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"Error generating Gemini response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_openai_response(self, query: str, context: str, search_strategy: str) -> str:
        """Generate response using OpenAI GPT."""
        try:
            if not self.openai_key:
                return "Error: OpenAI API key not configured. Please set your OpenAI API key in config.json or environment variables."
            
            prompt = f"""You are an AI assistant that helps users find information from podcast transcripts. 
Based on the provided context from podcast episodes, answer the user's question accurately and provide specific timestamps when possible.

Context from podcast transcripts:
{context}

User Question: {query}

Instructions:
1. Answer based only on the provided context
2. Include specific episode titles and timestamps in your response
3. If the context doesn't contain enough information, say so
4. Provide direct quotes when relevant
5. Format timestamps as MM:SS
6. Mention speakers when relevant
7. If searching across multiple episodes, highlight patterns or differences

Answer:"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes podcast content and provides accurate, timestamped responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating OpenAI response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format."""
        try:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        except:
            return "00:00"
    
    def get_search_recommendations(self, query: str) -> Dict[str, Any]:
        """Get search strategy recommendations based on query type."""
        try:
            # Simple keyword-based strategy recommendation
            query_lower = query.lower()
            
            # Keywords that suggest semantic search
            semantic_keywords = ['how', 'why', 'what', 'explain', 'describe', 'discuss', 'analyze']
            # Keywords that suggest keyword search
            keyword_keywords = ['specific', 'exact', 'precise', 'term', 'phrase']
            
            semantic_score = sum(1 for word in semantic_keywords if word in query_lower)
            keyword_score = sum(1 for word in keyword_keywords if word in query_lower)
            
            if semantic_score > keyword_score:
                primary_recommendation = "semantic"
            elif keyword_score > semantic_score:
                primary_recommendation = "keyword"
            else:
                primary_recommendation = "hybrid"
            
            all_strategies = {
                "semantic": "Best for conceptual questions and understanding context",
                "keyword": "Best for finding specific terms and exact matches",
                "hybrid": "Combines semantic understanding with keyword precision"
            }
            
            return {
                "primary_recommendation": primary_recommendation,
                "all_strategies": all_strategies
            }
            
        except Exception as e:
            logging.error(f"Error getting search recommendations: {e}")
            return {
                "primary_recommendation": "semantic",
                "all_strategies": {
                    "semantic": "Default strategy",
                    "keyword": "Keyword-based search",
                    "hybrid": "Combined approach"
                }
            }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available models with their status."""
        models = {
            "gemini": {
                "name": "Google Gemini",
                "description": "Google's advanced language model",
                "status": "available" if self.gemini_key else "no_api_key",
                "speed": "Fast",
                "accuracy": "Excellent"
            },
            "openai": {
                "name": "OpenAI GPT",
                "description": "OpenAI's GPT-3.5-turbo model",
                "status": "available" if self.openai_key else "no_api_key",
                "speed": "Medium",
                "accuracy": "Excellent"
            }
        }
        return models
