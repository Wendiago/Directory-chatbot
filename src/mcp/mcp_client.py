import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from mcp.client import ClientSession, StdioServerParameters
from mcp.types import Tool
from pathlib import Path

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self, server_command: str = None, server_args: List[str] = None):
        self.server_command = server_command or "python"
        self.server_args = server_args or [
            "-m", "chatbot.mcp.file_system_server"
        ]
        self.session: Optional[ClientSession] = None
        self.tools: List[Tool] = []
    
    async def connect(self) -> bool:
        """Connect to the MCP file system server"""
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args
            )
            
            # Create client session
            self.session = ClientSession(server_params)
            
            # Initialize the session
            await self.session.initialize()
            
            # List available tools
            tools_result = await self.session.list_tools()
            self.tools = tools_result.tools
            
            logger.info(f"Connected to MCP server with {len(self.tools)} tools available")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from MCP server")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Call a tool on the MCP server"""
        try:
            if not self.session:
                logger.error("Not connected to MCP server")
                return None
            
            # Call the tool
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content from result
            if result.content and len(result.content) > 0:
                return result.content[0].text
            else:
                return "No content returned"
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return None
    
    async def list_files(self, path: str = None) -> Optional[str]:
        """List files in a directory"""
        arguments = {}
        if path:
            arguments["path"] = path
        
        return await self.call_tool("list_files", arguments)
    
    async def read_file(self, path: str, encoding: str = "utf-8") -> Optional[str]:
        """Read a file's content"""
        arguments = {
            "path": path,
            "encoding": encoding
        }
        
        return await self.call_tool("read_file", arguments)
    
    async def get_file_metadata(self, path: str) -> Optional[str]:
        """Get metadata for a file or directory"""
        arguments = {"path": path}
        
        return await self.call_tool("get_file_metadata", arguments)
    
    async def extract_document_content(self, path: str) -> Optional[str]:
        """Extract content from a document file"""
        arguments = {"path": path}
        
        return await self.call_tool("extract_document_content", arguments)
    
    async def search_files(self, query: str, path: str = None, file_types: List[str] = None) -> Optional[str]:
        """Search for files by name or content"""
        arguments = {"query": query}
        
        if path:
            arguments["path"] = path
        
        if file_types:
            arguments["file_types"] = file_types
        
        return await self.call_tool("search_files", arguments)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.name for tool in self.tools]
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description of a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.description
        return None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

class MCPFileManager:
    """High-level file manager that combines MCP client with RAG operations"""
    
    def __init__(self, mcp_client: MCPClient, rag_manager=None):
        self.mcp_client = mcp_client
        self.rag_manager = rag_manager
    
    async def explore_directory(self, path: str = None) -> Dict[str, Any]:
        """Explore a directory and get file listing"""
        try:
            result = await self.mcp_client.list_files(path)
            return {
                "success": True,
                "path": path,
                "listing": result
            }
        except Exception as e:
            logger.error(f"Error exploring directory {path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive information about a file"""
        try:
            # Get metadata
            metadata_result = await self.mcp_client.get_file_metadata(file_path)
            
            # Try to extract content if it's a supported document
            content = None
            if self._is_supported_document(file_path):
                content = await self.mcp_client.extract_document_content(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "metadata": metadata_result,
                "content_preview": content[:500] + "..." if content and len(content) > 500 else content
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_and_index(self, query: str, search_path: str = None) -> Dict[str, Any]:
        """Search for files and optionally index them for RAG"""
        try:
            # Search for files
            search_result = await self.mcp_client.search_files(query, search_path)
            
            if not self.rag_manager:
                return {
                    "success": True,
                    "search_results": search_result,
                    "indexed": False,
                    "message": "RAG manager not available for indexing"
                }
            
            # Parse search results to get file paths
            file_paths = self._extract_file_paths_from_search(search_result)
            
            # Index found files
            indexed_files = []
            for file_path in file_paths:
                if self._is_supported_document(file_path):
                    index_result = self.rag_manager.process_and_index_file(file_path)
                    if index_result["success"]:
                        indexed_files.append(file_path)
            
            return {
                "success": True,
                "search_results": search_result,
                "indexed_files": indexed_files,
                "total_indexed": len(indexed_files)
            }
            
        except Exception as e:
            logger.error(f"Error in search and index: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _is_supported_document(self, file_path: str) -> bool:
        """Check if file is a supported document type"""
        from config.settings import Settings
        return Path(file_path).suffix.lower() in Settings.SUPPORTED_EXTENSIONS
    
    def _extract_file_paths_from_search(self, search_result: str) -> List[str]:
        """Extract file paths from search result text"""
        # This is a simple parser - in practice you might want more sophisticated parsing
        file_paths = []
        lines = search_result.split('\n')
        
        for line in lines:
            if line.strip().startswith('📄'):
                # Extract path from line like "📄 /path/to/file.txt"
                path_part = line.strip()[2:].strip()  # Remove emoji and whitespace
                if path_part and Path(path_part).exists():
                    file_paths.append(path_part)
        
        return file_paths
    
    async def get_document_content(self, file_path: str) -> Dict[str, Any]:
        """Get full content of a document file"""
        try:
            if not self._is_supported_document(file_path):
                return {
                    "success": False,
                    "error": "File type not supported for content extraction"
                }
            
            content = await self.mcp_client.extract_document_content(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content
            }
            
        except Exception as e:
            logger.error(f"Error getting document content for {file_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            } 