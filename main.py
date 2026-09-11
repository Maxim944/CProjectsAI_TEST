from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="ATIG AI Agent Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    user_id: str = "default_user"
    message: str

@app.get("/")
async def root():
    return {"status": "online", "system": "ATIG AI Core Active"}

@app.post("/api/chat")
async def chat(request: PromptRequest):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5-coder:7b",
                    "prompt": request.message,
                    "stream": False
                }
            )
            data = response.json()
            return {"reply": data.get("response", "Ошибка получения ответа")}
    except Exception as e:
        return {"reply": f"Ошибка соединения с Ollama: {str(e)}"}