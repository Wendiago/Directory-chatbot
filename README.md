## THIS IS A MCP DIRECTORY CHATBOT PROJECT THAT USES "https://github.com/keli-wen/mcp_chatbot" FOR THE BASE

#Installation steps:
- requires-python = ">=3.10"
- Make sure to have ollama installed in your local first

1. Pull qwen3:8b model, we'll be using this model: ollama pull qwen3:8b
2. Clone the repo to local
3. Create .venv: uv venv .venv --python=3.10
4. Activate .venv
5. Install dependencies from requirements.txt: pip install -r requirements.txt or with uv: uv pip install -r requirements.txt
6. For env, you just need these two: OLLAMA_MODEL_NAME="qwen3:8b", OLLAMA_BASE_URL="http://localhost:11434"
7. MCP File system server config: 
Modify your mcp_servers/servers_config.json as this example
- command is the path to your uv, you can use "which uv" to find it
- second arg is absolute path to our mcp_servers on your local computer
- last arg is the root folder for file system server, which is the boundary for file system server
{
  "mcpServers": {
    "file_system": {
      "command": "C:\\Users\\LTC\\.local\\bin\\uv.exe",
      "args": [
        "--directory",
        "E:\\HoangNgoc\\NewTechInSE\\chatbot\\mcp_servers",
        "run",
        "file_system.py",
        "E:\\HoangNgoc\\NewTechInSE\\project3\\data"
      ]
    }
  }
}
8. Run chat bot: streamlit run ./example/chatbot_streamlit/app.py

