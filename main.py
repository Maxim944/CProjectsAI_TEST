import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

app = FastAPI(title="ATIG AI Core - Local Autonomous Agent")

# Настройка CORS для работы веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение к локальной Ollama через эмуляцию OpenAI API
client = openai.OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"  # Фиктивный ключ для локального подключения
)

# Хранилище долговременной памяти (Long-Term Memory)
class UserMemory:
    def __init__(self):
        self.profile_data: Dict[str, Any] = {
            "user_id": "usr_001",
            "preferences": "Предпочитает краткие отчеты, работает в сферах IT и безопасности",
            "active_tasks": []
        }

    def get_context(self) -> str:
        return f"Профиль пользователя: {json.dumps(self.profile_data, ensure_ascii=False)}"

    def update_task(self, task_desc: str):
        self.profile_data["active_tasks"].append(task_desc)

memory = UserMemory()

# Инструменты агента (Tool Calling)
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_task_action",
            "description": "Автоматически создает и фиксирует задачу в плане пользователя",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_title": {"type": "string", "description": "Название задачи"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["task_title", "priority"]
            }
        }
    }
]

class AgentRequest(BaseModel):
    user_prompt: str

@app.post("/api/v1/agent/run")
async def run_agent(request: AgentRequest):
    try:
        system_prompt = (
            "Ты — автономный ИИ-агент (Agentic AI ATIG). Твоя цель — самостоятельно анализировать "
            "запрос пользователя, использовать память и вызывать необходимые инструменты для выполнения задач.\n"
            f"{memory.get_context()}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_prompt}
        ]

        # Запрос к локальной Qwen 2.5 Coder через Ollama
        response = client.chat.completions.create(
            model="qwen2.5-coder:7b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "create_task_action":
                    memory.update_task(function_args.get("task_title"))
                    
                    messages.append(response_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"status": "success", "added_task": function_args.get("task_title")})
                    })

            final_response = client.chat.completions.create(
                model="qwen2.5-coder:7b",
                messages=messages
            )
            return {"status": "completed", "agent_response": final_response.choices[0].message.content}

        return {"status": "completed", "agent_response": response_message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))