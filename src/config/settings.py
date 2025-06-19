from pathlib import Path

class Settings:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    METADATA_DIR = DATA_DIR / "metadata"
    CLASSIFICATIONS_DIR = DATA_DIR / "classifications"
    
    # Ollama configuration
    OLLAMA_HOST = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:8b" 
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.txt', '.md']
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIEVED_DOCS = 5
    
    # Classification categories
    FILE_CATEGORIES = {
        "A": "Planning and Strategy Documents",
        "B": "Marketing and Sales Materials", 
        "C": "Technical Documentation",
        "D": "Reports and Analysis",
        "E": "Presentations",
        "F": "Other Documents"
    }
    
    # MCP Server settings
    MCP_SERVER_HOST = "localhost"
    MCP_SERVER_PORT = 8000
    MCP_CLOUD_API_URL = "https://your-mcp-cloud-api.com"
    
    # UI settings
    CHAT_TITLE = "Local AI Document Assistant"
    MAX_CHAT_HISTORY = 50
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATA_DIR, cls.DOCUMENTS_DIR, 
                        cls.EMBEDDINGS_DIR, cls.METADATA_DIR, 
                        cls.CLASSIFICATIONS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)