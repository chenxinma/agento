# Pydantic-AI Agent OpenAI API Wrapper

This project wraps a pydantic-ai agent as an OpenAI API-compatible service, allowing you to use your pydantic-ai agents with existing OpenAI client libraries and tools.

## Overview

The service reimplements the OpenAI API interface standards to expose a pydantic-ai agent as if it were a large language model service. This enables seamless integration with existing applications that expect OpenAI's API format.

## Features

- OpenAI API-compatible endpoints
- Pydantic-ai agent integration
- RESTful API interface
- Drop-in replacement for OpenAI API clients
- API key authentication support
- Session-based request tracking
- Streaming and non-streaming responses

## Usage

Clients can interact with this service using the same interface as OpenAI's API, making it compatible with existing OpenAI client libraries and tools.

### API Endpoints

- `GET /` - Health check
- `GET /v1/models` - List available models (returns single "pydantic-ai-agent" model)
- `POST /v1/chat/completions` - Create chat completions (OpenAI-compatible)

### Authentication

The service supports optional API key authentication. Set the `API_KEY` environment variable to enable authentication:

```bash
export API_KEY="your-secret-key"
```

When authentication is enabled, include the API key in the `Authorization` header:

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:8000/v1/models
```

### Configuration

Environment variables:
- `API_KEY`: Optional API key for authentication
- `AGENTO_SALT`: Optional salt for session ID generation (defaults to "abc")

## Project Structure

- `pyproject.toml` - Project configuration and dependencies
- `src/main.py` - Demo Application entry point wrapper