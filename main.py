import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

app = FastAPI(title="ATIG - Autonomous Agent")

# Разрешаем кросс-доменные запросы (CORS) для подключения с GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Используем совместимый с OpenAI бесплатный API Groq
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_ВАШ_КЛЮЧ_GROQ"  # Вставьте сюда ваш ключ gsk_...
)

# История сообщений для поддержки контекста беседы
conversation_history: List[Dict[str, str]] = []

class UserMemory:
    def __init__(self):
        self.profile_data: Dict[str, Any] = {
            "user_id": "usr_001",
            "name": "Максим",
            "preferences": "Предпочитает краткие отчеты, работает в сферах IT и безопасности",
            "active_tasks": []
        }

    def get_context(self) -> str:
        return f"Профиль пользователя: {json.dumps(self.profile_data, ensure_ascii=False)}"

    def update_task(self, task_desc: str):
        self.profile_data["active_tasks"].append(task_desc)

memory = UserMemory()

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
            "Ты — автономный ИИ-агент ATIG. Твоя цель — помогать пользователю, "
            "используя память и вызывая инструменты при необходимости. "
            "Отвечай естественным текстом, вежливо и по существу.\n"
            f"{memory.get_context()}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        # Передаем последние 6 сообщений контекста
        messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": request.user_prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "create_task_action":
                    task_title = function_args.get("task_title")
                    memory.update_task(task_title)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps({"status": "success", "message": f"Задача '{task_title}' успешно добавлена."})
                    })

            # Запрашиваем финальный текст у модели после исполнения функции
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            final_text = final_response.choices[0].message.content
        else:
            final_text = response_message.content

        # Запоминаем шаг диалога
        conversation_history.append({"role": "user", "content": request.user_prompt})
        conversation_history.append({"role": "assistant", "content": final_text})

        return {"status": "completed", "agent_response": final_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))