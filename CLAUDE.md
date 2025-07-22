# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Pydantic-AI Agent OpenAI API Wrapper that exposes a pydantic-ai agent as an OpenAI API-compatible service. It provides RESTful endpoints that match OpenAI's API format, allowing existing OpenAI client libraries to work seamlessly with pydantic-ai agents.

## Architecture
- **FastAPI-based**: RESTful API server built with FastAPI
- **OpenAI API Compatible**: Implements `/v1/chat/completions` and `/v1/models` endpoints
- **Pydantic-AI Integration**: Uses `pydantic-ai` library for agent orchestration
- **Streaming Support**: Supports both streaming and non-streaming responses

## Key Components
- **main.py**: Single-file FastAPI application containing all endpoints and logic
- **OpenAI API Models**: Pydantic models matching OpenAI's API schema (`ChatMessage`, `ChatCompletionRequest`, etc.)
- **Agent Wrapper**: Uses `pydantic-ai` Agent with GPT-4o-mini backend
- **Streaming Response**: Async generator for SSE streaming responses

## Development Commands

### Setup & Installation
```bash
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

### Running the Service
```bash
# Development server
python main.py

# With uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing
```bash
# Basic health check
curl http://localhost:8000/

# List available models
curl http://localhost:8000/v1/models

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "pydantic-ai-agent",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### API Endpoints
- `GET /` - Health check
- `GET /v1/models` - List available models (returns single "pydantic-ai-agent" model)
- `POST /v1/chat/completions` - Create chat completions (OpenAI-compatible)

## Configuration
- **Port**: 8000 (default)
- **Model**: Uses OpenAI GPT-4o-mini via pydantic-ai
- **Dependencies**: FastAPI, uvicorn, pydantic, pydantic-ai

## Python Version
Requires Python >=3.11 as specified in pyproject.toml