import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from config.settings import Settings

logger = logging.getLogger(__name__)

class RAGManager:
    def __init__(self, vector_store_path: str = None):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore(vector_store_path)
        self.processed_files = set()
    
    def process_and_index_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Process and index all documents in a directory"""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                return {"success": False, "error": f"Directory does not exist: {directory_path}"}
            
            logger.info(f"Processing directory: {directory_path}")
            
            # Process documents
            documents = self.document_processor.process_directory(directory, recursive)
            
            if not documents:
                return {"success": False, "error": "No documents found to process"}
            
            # Add to vector store
            success = self.vector_store.add_documents(documents)
            
            if success:
                # Track processed files
                for doc in documents:
                    self.processed_files.add(doc['metadata']['file_path'])
                
                stats = self.vector_store.get_collection_stats()
                
                return {
                    "success": True,
                    "documents_processed": len(documents),
                    "files_processed": len(set(doc['metadata']['file_path'] for doc in documents)),
                    "collection_stats": stats
                }
            else:
                return {"success": False, "error": "Failed to add documents to vector store"}
                
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def process_and_index_file(self, file_path: str) -> Dict[str, Any]:
        """Process and index a single file"""
        try:
            file = Path(file_path)
            if not file.exists():
                return {"success": False, "error": f"File does not exist: {file_path}"}
            
            logger.info(f"Processing file: {file_path}")
            
            # Process document
            documents = self.document_processor.process_file(file)
            
            if not documents:
                return {"success": False, "error": "No content extracted from file"}
            
            # Remove existing documents for this file
            self.vector_store.delete_documents_by_file(file_path)
            
            # Add new documents
            success = self.vector_store.add_documents(documents)
            
            if success:
                self.processed_files.add(file_path)
                
                return {
                    "success": True,
                    "documents_processed": len(documents),
                    "file_path": file_path
                }
            else:
                return {"success": False, "error": "Failed to add documents to vector store"}
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def search_documents(self, query: str, n_results: int = None, filter_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for documents using RAG"""
        try:
            logger.info(f"Searching for: {query}")
            
            # Search vector store
            results = self.vector_store.search(query, n_results, filter_metadata)
            
            if not results:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No relevant documents found"
                }
            
            # Format results for better presentation
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result['content'],
                    "file_name": result['metadata'].get('file_name', 'Unknown'),
                    "file_path": result['metadata'].get('file_path', 'Unknown'),
                    "chunk_id": result['metadata'].get('chunk_id', 0),
                    "similarity_score": round(result['similarity_score'], 3),
                    "rank": result['rank']
                })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return {"success": False, "error": str(e)}
    
    def get_document_context(self, query: str, max_chunks: int = 3) -> str:
        """Get relevant document context for a query"""
        try:
            results = self.vector_store.search(query, max_chunks)
            
            if not results:
                return ""
            
            # Build context from top results
            context_parts = []
            for result in results:
                context_parts.append(
                    f"Document: {result['metadata'].get('file_name', 'Unknown')}\n"
                    f"Content: {result['content']}\n"
                    f"---\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return ""
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents"""
        try:
            stats = self.vector_store.get_collection_stats()
            stats['processed_files'] = len(self.processed_files)
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear all indexed documents"""
        try:
            success = self.vector_store.clear_collection()
            if success:
                self.processed_files.clear()
                return {"success": True, "message": "Index cleared successfully"}
            else:
                return {"success": False, "error": "Failed to clear index"}
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            return {"success": False, "error": str(e)}
    
    def update_document(self, file_path: str) -> Dict[str, Any]:
        """Update a document in the index"""
        return self.process_and_index_file(file_path)
    
    def remove_document(self, file_path: str) -> Dict[str, Any]:
        """Remove a document from the index"""
        try:
            success = self.vector_store.delete_documents_by_file(file_path)
            if success:
                self.processed_files.discard(file_path)
                return {"success": True, "message": f"Document {file_path} removed from index"}
            else:
                return {"success": False, "error": f"Failed to remove document {file_path}"}
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_similar_documents(self, document_content: str, n_results: int = 5) -> Dict[str, Any]:
        """Find documents similar to given content"""
        try:
            results = self.vector_store.search_similar_documents(document_content, n_results)
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result['content'],
                    "file_name": result['metadata'].get('file_name', 'Unknown'),
                    "file_path": result['metadata'].get('file_path', 'Unknown'),
                    "similarity_score": round(result['similarity_score'], 3),
                    "rank": result['rank']
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return {"success": False, "error": str(e)}
    
    def is_file_indexed(self, file_path: str) -> bool:
        """Check if a file is already indexed"""
        return file_path in self.processed_files
    
    def get_indexed_files(self) -> List[str]:
        """Get list of all indexed files"""
        return list(self.processed_files) 