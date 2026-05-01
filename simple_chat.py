"""
Created on Wed Apr 29 18:07:24 2026

@author: rublev.an
"""
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.prompt import Prompt


# ============ КОНФИГУРАЦИЯ ============

# Настройки подключения к vLLM (из вашего конфига)
VLLM_BASE_URL = "<your-LLM-base-url>"
VLLM_API_KEY = "<your-api-secret-key>"
MODEL_NAME = "<your-LLM-model-name>"

# Температура генерации (0.0 — детерминированно, 1.0 — креативно)
TEMPERATURE = 0.0

# Максимальное количество итераций агента (защита от бесконечных циклов)
MAX_ITERATIONS = 10

# ============ ИНИЦИАЛИЗАЦИЯ ============

console = Console()

def create_llm():
    """Создание LLM-клиента для vLLM"""
    extra_body =  {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=4096,  # Максимальное количество токенов в ответе
        timeout=120,  # Таймаут запроса в секундах (для длинных ответов)
        max_retries=3,  # Количество повторных попыток при ошибке
        extra_body=extra_body,
    )

def simple_chat():
    """Простой чат-бот без агента"""
    llm = create_llm()
    
    messages = [
        ("system", "Ты — AI ассистент по закупкам. Отвечай кратко и по делу.")
    ]
    
    console.print("[green]Простой LLM чат-бот (без инструментов)[/green]")
    console.print("[dim]Введите /exit для выхода[/dim]\n")
    
    while True:
        user_input = Prompt.ask("[bold cyan]Вы[/bold cyan]")
        
        if user_input.lower() in ['/exit', '/quit']:
            break
        
        messages.append(("human", user_input))
        
        with console.status("[cyan]Генерация...[/cyan]"):
            response = llm.invoke(messages)
        
        messages.append(("ai", response.content))
        
        console.print(f"[bold green]Бот[/bold green]: {response.content}\n")
        
if __name__ == "__main__":
    simple_chat()
