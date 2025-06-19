#!/usr/bin/env python3
"""
Startup script for the RAG Chatbot Streamlit app
"""

import sys
import os
from pathlib import Path

# Add the chatbot directory to Python path
chatbot_dir = Path(__file__).parent / "chatbot"
sys.path.insert(0, str(chatbot_dir))

def main():
    """Main entry point for Streamlit app"""
    try:
        # Import and run the Streamlit app
        from ui.streamlit_app import main as streamlit_main
        streamlit_main()
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 