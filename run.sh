#!/bin/bash

## Running streamlit with logging from MCP server
echo "Starting MCP Server logging in background..."

# Create named pipe for MCP logs
mkfifo /tmp/mcp_logs.pipe 2>/dev/null || true

# Start log monitoring in background
tail -f /tmp/mcp_logs.pipe &
LOG_PID=$!

# Export environment variables for MCP server logging
export MCP_LOG_LEVEL=DEBUG
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Start Streamlit with logging
echo "Starting Streamlit app..."
streamlit run ./example/chatbot_streamlit/app.py 2>&1 | tee -a /tmp/mcp_logs.pipe

# Cleanup
kill $LOG_PID 2>/dev/null || true
rm -f /tmp/mcp_logs.pipe