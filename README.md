# Pydantic-AI Agent OpenAI API Wrapper

This project wraps a pydantic-ai agent as an OpenAI API-compatible service, allowing you to use your pydantic-ai agents with existing OpenAI client libraries and tools.

## Overview

The service reimplements the OpenAI API interface standards to expose a pydantic-ai agent as if it were a large language model service. This enables seamless integration with existing applications that expect OpenAI's API format.

## Features

- OpenAI API-compatible endpoints
- Pydantic-ai agent integration
- RESTful API interface
- Drop-in replacement for OpenAI API clients

## Usage

Clients can interact with this service using the same interface as OpenAI's API, making it compatible with existing OpenAI client libraries and tools.

### API Endpoints

- `GET /` - Health check
- `GET /v1/models` - List available models (returns single "pydantic-ai-agent" model)
- `POST /v1/chat/completions` - Create chat completions (OpenAI-compatible)

## Project Structure

- `main.py` - Main application entry point
- `pyproject.toml` - Project configuration and dependencies