import os
import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import json

class TextIndexer:
    """Manages text indexing and search functionality for podcast transcripts."""
    
    def __init__(self):
        """Initialize the text indexer with ChromaDB."""
        db_path = os.path.abspath(os.path.join(".", "chroma_db"))
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use a single embedding model for simplicity
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collection for podcast chunks
        self.collection = self.client.get_or_create_collection(
            name="podcast_chunks",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        logging.info("Text indexer initialized successfully")

    def add_transcript_chunks(self, chunks: List[Dict], episode_id: str, episode_title: str) -> None:
        """Add transcript chunks to the search index."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Create document text
                text = chunk.get('text', '')
                if not text.strip():
                    continue
                
                # Create metadata
                metadata = {
                    'episode_id': episode_id,
                    'episode_title': episode_title,
                    'start_time': chunk.get('start_time', 0),
                    'end_time': chunk.get('end_time', 0),
                    'speaker': chunk.get('speaker', 'Unknown'),
                    'chunk_index': i
                }
                
                # Create unique ID
                chunk_id = f"{episode_id}_chunk_{i}"
                
                documents.append(text)
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logging.info(f"Added {len(documents)} chunks for episode '{episode_title}'")
            else:
                logging.warning(f"No valid chunks found for episode '{episode_title}'")
                
        except Exception as e:
            logging.error(f"Error adding transcript chunks: {e}")
            raise
    
    def search_similar_content(self, query: str, n_results: int = 5, 
                             search_strategy: str = "semantic") -> Dict[str, Any]:
        """Search for similar content using different strategies."""
        try:
            if search_strategy == "semantic":
                return self._semantic_search(query, n_results)
            elif search_strategy == "keyword":
                return self._keyword_search(query, n_results)
            elif search_strategy == "hybrid":
                return self._hybrid_search(query, n_results)
            else:
                return self._semantic_search(query, n_results)
                
        except Exception as e:
            logging.error(f"Error searching content: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def _semantic_search(self, query: str, n_results: int) -> Dict[str, Any]:
        """Perform semantic search using embeddings."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            return results
        except Exception as e:
            logging.error(f"Semantic search error: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def _keyword_search(self, query: str, n_results: int) -> Dict[str, Any]:
        """Perform keyword-based search."""
        try:
            # Simple keyword matching - could be enhanced with more sophisticated keyword extraction
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results for filtering
                include=["documents", "metadatas", "distances"]
            )
            
            # Filter results based on keyword presence
            filtered_documents = []
            filtered_metadatas = []
            filtered_distances = []
            
            query_terms = query.lower().split()
            
            for i, doc in enumerate(results.get('documents', [[]])[0]):
                doc_lower = doc.lower()
                # Check if any query term is in the document
                if any(term in doc_lower for term in query_terms):
                    filtered_documents.append(doc)
                    filtered_metadatas.append(results.get('metadatas', [[]])[0][i])
                    filtered_distances.append(results.get('distances', [[]])[0][i])
                
                if len(filtered_documents) >= n_results:
                    break
            
            return {
                "documents": [filtered_documents],
                "metadatas": [filtered_metadatas],
                "distances": [filtered_distances]
            }
            
        except Exception as e:
            logging.error(f"Keyword search error: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def _hybrid_search(self, query: str, n_results: int) -> Dict[str, Any]:
        """Perform hybrid search combining semantic and keyword approaches."""
        try:
            # Get semantic results
            semantic_results = self._semantic_search(query, n_results)
            
            # Get keyword results
            keyword_results = self._keyword_search(query, n_results)
            
            # Combine and rank results
            combined_documents = []
            combined_metadatas = []
            combined_distances = []
            
            # Add semantic results first (higher weight)
            semantic_docs = semantic_results.get('documents', [[]])[0]
            semantic_metas = semantic_results.get('metadatas', [[]])[0]
            semantic_dists = semantic_results.get('distances', [[]])[0]
            
            for i, doc in enumerate(semantic_docs):
                if doc not in combined_documents:
                    combined_documents.append(doc)
                    combined_metadatas.append(semantic_metas[i])
                    combined_distances.append(semantic_dists[i])
            
            # Add keyword results
            keyword_docs = keyword_results.get('documents', [[]])[0]
            keyword_metas = keyword_results.get('metadatas', [[]])[0]
            keyword_dists = keyword_results.get('distances', [[]])[0]
            
            for i, doc in enumerate(keyword_docs):
                if doc not in combined_documents and len(combined_documents) < n_results:
                    combined_documents.append(doc)
                    combined_metadatas.append(keyword_metas[i])
                    combined_distances.append(keyword_dists[i])
            
            return {
                "documents": [combined_documents[:n_results]],
                "metadatas": [combined_metadatas[:n_results]],
                "distances": [combined_distances[:n_results]]
            }
            
        except Exception as e:
            logging.error(f"Hybrid search error: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def search_by_episode(self, episode_id: str, query: str = "", n_results: int = 10) -> Dict[str, Any]:
        """Search within a specific episode."""
        try:
            where_clause = {"episode_id": episode_id}
            
            if query:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # Get all chunks from the episode
                results = self.collection.get(
                    where=where_clause,
                    include=["documents", "metadatas"]
                )
                # Convert to expected format
                results = {
                    "documents": [results.get('documents', [])],
                    "metadatas": [results.get('metadatas', [])],
                    "distances": [[0.0] * len(results.get('documents', []))]
                }
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching by episode: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed content."""
        try:
            count = self.collection.count()
            
            # Get unique episodes
            all_metadata = self.collection.get(include=["metadatas"])
            episode_ids = set()
            episode_titles = {}
            for metadata in all_metadata.get('metadatas', []):
                if metadata and 'episode_id' in metadata:
                    episode_ids.add(metadata['episode_id'])
                    if 'episode_title' in metadata:
                        episode_titles[metadata['episode_id']] = metadata['episode_title']
            
            # Calculate capacity info
            capacity_status = "Ready for more episodes"
            if len(episode_ids) >= 3:
                capacity_status = f"Successfully indexing {len(episode_ids)} episodes"
            elif len(episode_ids) == 2:
                capacity_status = "Ready for 3rd episode"
            elif len(episode_ids) == 1:
                capacity_status = "Ready for 2nd and 3rd episodes"
            
            return {
                "total_chunks": count,
                "total_episodes": len(episode_ids),
                "episode_ids": list(episode_ids),
                "episode_titles": episode_titles,
                "indexing_capacity": capacity_status,
                "max_capacity": 100  # Maximum episodes allowed
            }
            
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "total_episodes": 0, "episode_ids": [], "indexing_capacity": "Ready for 3+ episodes"}
    
    def clear_collection(self) -> None:
        """Clear all indexed content."""
        try:
            self.collection.delete()
            logging.info("Collection cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing collection: {e}")
