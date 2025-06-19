#!/usr/bin/env python3
"""
Main entry point for the RAG Chatbot with MCP File System Integration
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from core.llm_client import LLMClient
from rag.rag_manager import RAGManager
from mcp.mcp_client import MCPClient, MCPFileManager
from config.settings import Settings
from config.model_config import ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self):
        self.llm_client = None
        self.rag_manager = None
        self.mcp_client = None
        self.file_manager = None
        self.conversation_history = []
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing RAG Chatbot...")
        
        try:
            # Initialize LLM client
            logger.info("Initializing LLM client...")
            self.llm_client = LLMClient()
            self.llm_client.pull_model()
            
            # Initialize RAG manager
            logger.info("Initializing RAG manager...")
            self.rag_manager = RAGManager()
            
            # Initialize MCP client
            logger.info("Initializing MCP client...")
            self.mcp_client = MCPClient()
            await self.mcp_client.connect()
            
            # Initialize file manager
            self.file_manager = MCPFileManager(self.mcp_client, self.rag_manager)
            
            logger.info("All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.disconnect()
        logger.info("Cleanup completed")
    
    async def index_directory(self, directory_path: str) -> Dict[str, Any]:
        """Index a directory for RAG"""
        logger.info(f"Indexing directory: {directory_path}")
        
        result = self.rag_manager.process_and_index_directory(directory_path)
        
        if result["success"]:
            logger.info(f"Successfully indexed {result['documents_processed']} documents")
        else:
            logger.error(f"Failed to index directory: {result['error']}")
        
        return result
    
    async def search_documents(self, query: str) -> Dict[str, Any]:
        """Search documents using RAG"""
        logger.info(f"Searching documents for: {query}")
        
        result = self.rag_manager.search_documents(query)
        
        if result["success"]:
            logger.info(f"Found {result['total_results']} relevant documents")
        else:
            logger.error(f"Search failed: {result['error']}")
        
        return result
    
    async def explore_files(self, path: str = None) -> Dict[str, Any]:
        """Explore files using MCP"""
        logger.info(f"Exploring files in: {path or 'current directory'}")
        
        result = await self.file_manager.explore_directory(path)
        
        if result["success"]:
            logger.info("File exploration completed")
        else:
            logger.error(f"File exploration failed: {result['error']}")
        
        return result
    
    async def chat(self, user_input: str) -> str:
        """Generate chat response using RAG and LLM"""
        logger.info(f"Processing chat input: {user_input}")
        
        # Get relevant context from RAG
        context = ""
        if self.rag_manager:
            context = self.rag_manager.get_document_context(user_input)
        
        # Prepare system prompt
        system_prompt = ModelConfig.SYSTEM_PROMPTS["chat"]
        if context:
            system_prompt += f"\n\nRelevant document context:\n{context}"
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        try:
            response_stream = self.llm_client.chat(
                messages=self.conversation_history,
                stream=True
            )
            
            # Collect response
            full_response = ""
            for chunk in response_stream:
                full_response += chunk
            
            # Add assistant response to history
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            logger.info("Response generated successfully")
            return full_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    async def interactive_chat(self):
        """Run interactive chat session"""
        print("\n" + "="*60)
        print("🤖 RAG Chatbot with MCP File System Integration")
        print("="*60)
        print("Commands:")
        print("  /index <path>     - Index a directory for RAG")
        print("  /search <query>   - Search documents")
        print("  /explore <path>   - Explore files")
        print("  /stats           - Show system statistics")
        print("  /clear           - Clear chat history")
        print("  /quit            - Exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', 'exit', 'quit']:
                    print("Goodbye!")
                    break
                
                elif user_input.startswith('/index '):
                    path = user_input[7:].strip()
                    result = await self.index_directory(path)
                    if result["success"]:
                        print(f"✅ Indexed {result['documents_processed']} documents")
                    else:
                        print(f"❌ Error: {result['error']}")
                
                elif user_input.startswith('/search '):
                    query = user_input[8:].strip()
                    result = await self.search_documents(query)
                    if result["success"]:
                        print(f"🔍 Found {result['total_results']} documents:")
                        for i, doc in enumerate(result["results"][:3]):  # Show top 3
                            print(f"  {i+1}. {doc['file_name']} (Score: {doc['similarity_score']})")
                            print(f"     {doc['content'][:100]}...")
                    else:
                        print(f"❌ Error: {result['error']}")
                
                elif user_input.startswith('/explore '):
                    path = user_input[9:].strip()
                    result = await self.explore_files(path)
                    if result["success"]:
                        print("📁 Directory contents:")
                        print(result["listing"])
                    else:
                        print(f"❌ Error: {result['error']}")
                
                elif user_input == '/stats':
                    if self.rag_manager:
                        stats = self.rag_manager.get_collection_stats()
                        print("📊 System Statistics:")
                        for key, value in stats.items():
                            print(f"  {key}: {value}")
                    else:
                        print("❌ RAG manager not available")
                
                elif user_input == '/clear':
                    self.conversation_history = []
                    print("🗑️ Chat history cleared")
                
                else:
                    # Regular chat
                    print("Bot: ", end="", flush=True)
                    response = await self.chat(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Chatbot with MCP Integration")
    parser.add_argument("--index", type=str, help="Index a directory for RAG")
    parser.add_argument("--search", type=str, help="Search documents")
    parser.add_argument("--explore", type=str, help="Explore files")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat")
    
    args = parser.parse_args()
    
    chatbot = RAGChatbot()
    
    try:
        # Initialize
        if not await chatbot.initialize():
            logger.error("Failed to initialize chatbot")
            return
        
        # Handle command line arguments
        if args.index:
            result = await chatbot.index_directory(args.index)
            if result["success"]:
                print(f"✅ Indexed {result['documents_processed']} documents")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.search:
            result = await chatbot.search_documents(args.search)
            if result["success"]:
                print(f"🔍 Found {result['total_results']} documents:")
                for i, doc in enumerate(result["results"][:5]):
                    print(f"  {i+1}. {doc['file_name']} (Score: {doc['similarity_score']})")
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.explore:
            result = await chatbot.explore_files(args.explore)
            if result["success"]:
                print("📁 Directory contents:")
                print(result["listing"])
            else:
                print(f"❌ Error: {result['error']}")
        
        elif args.interactive:
            await chatbot.interactive_chat()
        
        else:
            # Default to interactive mode
            await chatbot.interactive_chat()
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
    
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 