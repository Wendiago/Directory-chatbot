import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from config.settings import Settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or str(Settings.EMBEDDINGS_DIR)
        self.embedding_model = Settings.EMBEDDING_MODEL
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store initialized at {self.persist_directory}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents to add")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = f"{doc['metadata']['file_path']}_{doc['metadata']['chunk_id']}"
                ids.append(doc_id)
                texts.append(doc['content'])
                metadatas.append(doc['metadata'])
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def search(self, query: str, n_results: int = None, filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            n_results = n_results or Settings.MAX_RETRIEVED_DOCS
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'rank': i + 1
                    })
            
            logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results['documents'] and results['documents'][0]:
                return {
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID {doc_id}: {e}")
            return None
    
    def delete_documents_by_file(self, file_path: str) -> bool:
        """Delete all documents from a specific file"""
        try:
            # Get all documents for the file
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents for file {file_path}")
                return True
            
            logger.info(f"No documents found for file {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents for file {file_path}: {e}")
            return False
    
    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]) -> bool:
        """Update a specific document"""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[new_content],
                metadatas=[new_metadata]
            )
            logger.info(f"Updated document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample documents to analyze
            sample_results = self.collection.get(limit=100)
            
            # Calculate average document length
            if sample_results['documents']:
                avg_length = np.mean([len(doc) for doc in sample_results['documents']])
            else:
                avg_length = 0
            
            # Get unique file paths
            unique_files = set()
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    if metadata and 'file_path' in metadata:
                        unique_files.add(metadata['file_path'])
            
            return {
                'total_documents': count,
                'average_document_length': round(avg_length, 2),
                'unique_files': len(unique_files),
                'collection_name': self.collection.name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.collection.delete(where={})
            logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_documents_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific file"""
        try:
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            documents = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    documents.append({
                        'content': doc,
                        'metadata': metadata or {},
                        'id': results['ids'][i] if results['ids'] else None
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents for file {file_path}: {e}")
            return []
    
    def search_similar_documents(self, document_content: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document content"""
        try:
            results = self.collection.query(
                query_texts=[document_content],
                n_results=n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,
                        'rank': i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return [] 