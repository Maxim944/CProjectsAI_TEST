# ATIG — Autonomous AI Agent Infrastructure

Персональный искусственный интеллект и автономный агент.

## Архитектура
- **Backend:** FastAPI, Uvicorn, Async HTTP Client (`httpx`)
- **LLM Engine:** Local Ollama (`qwen2.5-coder:7b`)
- **Frontend:** HTML5, JS, CSS

## Быстрый запуск

1. Запустить локальный сервер моделей:
```bash
ollama run qwen2.5-coder:7b