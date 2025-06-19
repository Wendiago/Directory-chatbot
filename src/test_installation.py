#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import asyncio
from pathlib import Path

# Add the chatbot directory to Python path
chatbot_dir = Path(__file__).parent / "chatbot"
sys.path.insert(0, str(chatbot_dir))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from config.settings import Settings
        print("✅ Settings imported successfully")
        
        from config.model_config import ModelConfig
        print("✅ Model config imported successfully")
        
        from core.llm_client import LLMClient
        print("✅ LLM client imported successfully")
        
        from rag.document_processor import DocumentProcessor
        print("✅ Document processor imported successfully")
        
        from rag.vector_store import VectorStore
        print("✅ Vector store imported successfully")
        
        from rag.rag_manager import RAGManager
        print("✅ RAG manager imported successfully")
        
        from mcp.file_system_server import FileSystemServer
        print("✅ MCP file system server imported successfully")
        
        from mcp.mcp_client import MCPClient
        print("✅ MCP client imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_settings():
    """Test settings configuration"""
    print("\n🔍 Testing settings...")
    
    try:
        from config.settings import Settings
        
        # Test directory creation
        Settings.create_directories()
        print("✅ Directories created successfully")
        
        # Test settings access
        print(f"✅ Ollama host: {Settings.OLLAMA_HOST}")
        print(f"✅ Ollama model: {Settings.OLLAMA_MODEL}")
        print(f"✅ Embedding model: {Settings.EMBEDDING_MODEL}")
        print(f"✅ Supported extensions: {Settings.SUPPORTED_EXTENSIONS}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings error: {e}")
        return False

async def test_mcp_server():
    """Test MCP server functionality"""
    print("\n🔍 Testing MCP server...")
    
    try:
        from mcp.file_system_server import FileSystemServer
        
        # Create server instance
        server = FileSystemServer()
        print("✅ MCP server created successfully")
        
        # Test tool listing
        tools_result = await server.server.list_tools()
        print(f"✅ MCP server has {len(tools_result.tools)} tools")
        
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP server error: {e}")
        return False

def test_document_processor():
    """Test document processor"""
    print("\n🔍 Testing document processor...")
    
    try:
        from rag.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("✅ Document processor created successfully")
        
        # Test text cleaning
        test_text = "This   is   a   test   text   with   extra   spaces."
        cleaned = processor.clean_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test chunking
        test_text = "This is a longer text that should be chunked into smaller pieces. " * 10
        chunks = processor.chunk_text(test_text, chunk_size=50, chunk_overlap=10)
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

def test_vector_store():
    """Test vector store"""
    print("\n🔍 Testing vector store...")
    
    try:
        from rag.vector_store import VectorStore
        
        # Create vector store
        vector_store = VectorStore()
        print("✅ Vector store created successfully")
        
        # Test collection stats
        stats = vector_store.get_collection_stats()
        print(f"✅ Collection stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

async def test_llm_client():
    """Test LLM client (requires Ollama to be running)"""
    print("\n🔍 Testing LLM client...")
    
    try:
        from core.llm_client import LLMClient
        
        # Test connection
        client = LLMClient()
        print("✅ LLM client created successfully")
        
        # Test model availability
        try:
            client.pull_model()
            print("✅ Model check completed")
        except Exception as e:
            print(f"⚠️  Model check failed (Ollama may not be running): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting installation test...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Settings", test_settings),
        ("Document Processor", test_document_processor),
        ("Vector Store", test_vector_store),
        ("MCP Server", test_mcp_server),
        ("LLM Client", test_llm_client),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull Qwen model: ollama pull qwen2.5:7b")
        print("3. Run Streamlit app: streamlit run chatbot/ui/streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (requires 3.10+)")
        print("3. Ensure all required packages are installed")

if __name__ == "__main__":
    asyncio.run(main()) 