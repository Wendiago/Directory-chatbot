import streamlit as st
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Import our modules
from core.llm_client import LLMClient
from rag.rag_manager import RAGManager
from mcp.mcp_client import MCPClient, MCPFileManager
from config.settings import Settings
from config.model_config import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotUI:
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'rag_manager' not in st.session_state:
            st.session_state.rag_manager = None
        
        if 'mcp_client' not in st.session_state:
            st.session_state.mcp_client = None
        
        if 'file_manager' not in st.session_state:
            st.session_state.file_manager = None
        
        if 'llm_client' not in st.session_state:
            st.session_state.llm_client = None
        
        if 'current_directory' not in st.session_state:
            st.session_state.current_directory = str(Path.home())
        
        if 'indexed_files' not in st.session_state:
            st.session_state.indexed_files = set()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Settings.CHAT_TITLE,
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        """Initialize all components (LLM, RAG, MCP)"""
        try:
            # Initialize LLM client
            if not st.session_state.llm_client:
                with st.spinner("Initializing LLM client..."):
                    st.session_state.llm_client = LLMClient()
                    st.session_state.llm_client.pull_model()
            
            # Initialize RAG manager
            if not st.session_state.rag_manager:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag_manager = RAGManager()
            
            # Initialize MCP client
            if not st.session_state.mcp_client:
                with st.spinner("Initializing MCP client..."):
                    st.session_state.mcp_client = MCPClient()
                    # Note: MCP client connection will be handled in async context
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return False
    
    def render_sidebar(self):
        """Render the sidebar with file management options"""
        st.sidebar.title("📁 File Management")
        
        # Directory navigation
        st.sidebar.subheader("Directory Navigation")
        current_dir = st.text_input("Current Directory", value=st.session_state.current_directory)
        if current_dir != st.session_state.current_directory:
            st.session_state.current_directory = current_dir
        
        # File operations
        st.sidebar.subheader("File Operations")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📂 Explore Directory"):
                self.explore_directory()
        
        with col2:
            if st.button("🔍 Search Files"):
                self.show_search_interface()
        
        # RAG operations
        st.sidebar.subheader("RAG Operations")
        
        if st.button("📚 Index Directory"):
            self.index_directory()
        
        if st.button("🗑️ Clear Index"):
            self.clear_index()
        
        # Show stats
        if st.session_state.rag_manager:
            stats = st.session_state.rag_manager.get_collection_stats()
            if stats:
                st.sidebar.subheader("📊 Index Statistics")
                st.sidebar.metric("Total Documents", stats.get('total_documents', 0))
                st.sidebar.metric("Unique Files", stats.get('unique_files', 0))
                st.sidebar.metric("Avg Doc Length", stats.get('average_document_length', 0))
    
    def explore_directory(self):
        """Explore current directory using MCP"""
        async def _explore():
            try:
                async with MCPClient() as mcp_client:
                    file_manager = MCPFileManager(mcp_client, st.session_state.rag_manager)
                    result = await file_manager.explore_directory(st.session_state.current_directory)
                    
                    if result["success"]:
                        st.success("Directory explored successfully!")
                        st.text(result["listing"])
                    else:
                        st.error(f"Error exploring directory: {result['error']}")
                        
            except Exception as e:
                st.error(f"Error connecting to MCP server: {e}")
        
        asyncio.run(_explore())
    
    def show_search_interface(self):
        """Show file search interface"""
        st.subheader("🔍 File Search")
        
        search_query = st.text_input("Search query")
        search_path = st.text_input("Search path (optional)", value=st.session_state.current_directory)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Search Files"):
                if search_query:
                    self.search_files(search_query, search_path)
        
        with col2:
            if st.button("Search & Index"):
                if search_query:
                    self.search_and_index(search_query, search_path)
    
    def search_files(self, query: str, search_path: str = None):
        """Search for files using MCP"""
        async def _search():
            try:
                async with MCPClient() as mcp_client:
                    result = await mcp_client.search_files(query, search_path)
                    if result:
                        st.success("Search completed!")
                        st.text(result)
                    else:
                        st.error("Search failed")
                        
            except Exception as e:
                st.error(f"Error searching files: {e}")
        
        asyncio.run(_search())
    
    def search_and_index(self, query: str, search_path: str = None):
        """Search for files and index them"""
        async def _search_and_index():
            try:
                async with MCPClient() as mcp_client:
                    file_manager = MCPFileManager(mcp_client, st.session_state.rag_manager)
                    result = await file_manager.search_and_index(query, search_path)
                    
                    if result["success"]:
                        st.success(f"Search and index completed! Indexed {result['total_indexed']} files.")
                        st.text(result["search_results"])
                    else:
                        st.error(f"Error: {result['error']}")
                        
            except Exception as e:
                st.error(f"Error in search and index: {e}")
        
        asyncio.run(_search_and_index())
    
    def index_directory(self):
        """Index current directory for RAG"""
        if not st.session_state.rag_manager:
            st.error("RAG manager not initialized")
            return
        
        with st.spinner("Indexing directory..."):
            result = st.session_state.rag_manager.process_and_index_directory(
                st.session_state.current_directory
            )
            
            if result["success"]:
                st.success(f"Directory indexed successfully! Processed {result['documents_processed']} documents.")
                
                # Update indexed files
                stats = st.session_state.rag_manager.get_collection_stats()
                st.session_state.indexed_files = set(st.session_state.rag_manager.get_indexed_files())
            else:
                st.error(f"Error indexing directory: {result['error']}")
    
    def clear_index(self):
        """Clear the RAG index"""
        if not st.session_state.rag_manager:
            st.error("RAG manager not initialized")
            return
        
        if st.button("Confirm Clear Index"):
            result = st.session_state.rag_manager.clear_index()
            if result["success"]:
                st.success("Index cleared successfully!")
                st.session_state.indexed_files.clear()
            else:
                st.error(f"Error clearing index: {result['error']}")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🤖 " + Settings.CHAT_TITLE)
        
        # Chat input
        user_input = st.chat_input("Ask me anything about your documents...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Generate response
            self.generate_response(user_input)
    
    def generate_response(self, user_input: str):
        """Generate chatbot response using RAG and LLM"""
        try:
            # Get relevant context from RAG
            context = ""
            if st.session_state.rag_manager:
                context = st.session_state.rag_manager.get_document_context(user_input)
            
            # Prepare system prompt
            system_prompt = ModelConfig.SYSTEM_PROMPTS["chat"]
            if context:
                system_prompt += f"\n\nRelevant document context:\n{context}"
            
            # Generate response
            with st.spinner("Thinking..."):
                response_stream = st.session_state.llm_client.chat(
                    messages=st.session_state.chat_history,
                    stream=True
                )
                
                # Display response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in response_stream:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            
            # Add assistant response to history
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            
            # Trim history if too long
            if len(st.session_state.chat_history) > Settings.MAX_CHAT_HISTORY:
                st.session_state.chat_history = st.session_state.chat_history[-Settings.MAX_CHAT_HISTORY:]
                
        except Exception as e:
            st.error(f"Error generating response: {e}")
    
    def render_chat_history(self):
        """Render chat history"""
        st.subheader("💬 Chat History")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    def render_rag_search(self):
        """Render RAG search interface"""
        st.subheader("🔍 RAG Document Search")
        
        search_query = st.text_input("Search in indexed documents")
        
        if st.button("Search Documents") and search_query:
            if not st.session_state.rag_manager:
                st.error("RAG manager not initialized")
                return
            
            with st.spinner("Searching documents..."):
                result = st.session_state.rag_manager.search_documents(search_query)
                
                if result["success"]:
                    st.success(f"Found {result['total_results']} relevant documents")
                    
                    for i, doc in enumerate(result["results"]):
                        with st.expander(f"Document {i+1}: {doc['file_name']} (Score: {doc['similarity_score']})"):
                            st.write(f"**File:** {doc['file_path']}")
                            st.write(f"**Chunk:** {doc['chunk_id']}")
                            st.write(f"**Content:**")
                            st.text(doc['content'])
                else:
                    st.error(f"Search failed: {result['error']}")
    
    def run(self):
        """Main method to run the Streamlit app"""
        # Initialize components
        if not self.initialize_components():
            st.error("Failed to initialize components. Please check your configuration.")
            return
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Document Search", "📊 System Info"])
        
        with tab1:
            self.render_chat_interface()
            self.render_chat_history()
        
        with tab2:
            self.render_rag_search()
        
        with tab3:
            self.render_system_info()

def main():
    """Main entry point for the Streamlit app"""
    app = ChatbotUI()
    app.run()

if __name__ == "__main__":
    main() 