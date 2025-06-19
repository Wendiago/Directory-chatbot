from ollama import Options
class ModelConfig:
    # Ollama model parameters
    OLLAMA_PARAMS = Options(
        temperature=0.7,
        top_k= 40,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=4096,
        stop=["Ollama stop"]
    )
    
    # System prompts
    SYSTEM_PROMPTS = {
        "chat": """You are a helpful AI assistant specialized in document analysis and file management. 
You can search through documents, classify files, and provide detailed explanations of your reasoning process.
Always be clear and helpful in your responses.""",
        
        "classification": """You are a document classifier. Analyze the content and classify documents into appropriate categories:
A: Planning and Strategy Documents
B: Marketing and Sales Materials
C: Technical Documentation  
D: Reports and Analysis
E: Presentations
F: Other Documents

Provide your reasoning for the classification.""",
        
        "search": """You are helping to search through document contents. 
Analyze the user's query and provide relevant information from the documents.
Be specific about which documents contain the requested information."""
    }