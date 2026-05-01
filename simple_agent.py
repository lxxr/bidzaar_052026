"""
Created on Wed Apr 29 17:33:42 2026

@author: rublev.an
"""

#!/usr/bin/env python3
"""
CLI AI Agent на базе LangChain + vLLM
Запуск: python simple_agent.py
"""

import sys
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from datetime import datetime

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

# ============ ИНСТРУМЕНТЫ АГЕНТА ============

@tool
def get_current_time() -> str:
    """Получить текущее время и дату. Используй, когда нужно узнать точное время."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def read_file(filepath: str) -> str:
    """
    Прочитать содержимое файла.
    
    Args:
        filepath: Путь к файлу относительно текущей директории
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Содержимое файла {filepath}:\n{content}"
    except FileNotFoundError:
        return f"Ошибка: файл {filepath} не найден"
    except Exception as e:
        return f"Ошибка при чтении файла: {str(e)}"

@tool
def list_directory(path: str = ".") -> str:
    """
    Показать содержимое директории.
    
    Args:
        path: Путь к директории (по умолчанию текущая)
    """
    try:
        files = os.listdir(path)
        if not files:
            return f"Директория {path} пуста"
        return f"Содержимое {path}:\n" + "\n".join(f"  • {f}" for f in sorted(files))
    except Exception as e:
        return f"Ошибка: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """
    Вычислить математическое выражение.
    Поддерживает: +, -, *, /, **, (), скобки.
    
    Args:
        expression: Математическое выражение в виде строки (например, "2 + 2 * 3")
    """
    try:
        # Очищаем выражение от потенциально опасных конструкций
        allowed_chars = set("0123456789+-*/().%** ")
        if not all(c in allowed_chars for c in expression):
            return "Ошибка: выражение содержит недопустимые символы"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {expression} = {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

@tool
def word_count(text: str) -> str:
    """
    Подсчитать количество слов и символов в тексте.
    
    Args:
        text: Текст для анализа
    """
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return f"Слов: {len(words)}, Символов (всего): {chars}, Символов (без пробелов): {chars_no_spaces}"

# Список всех доступных инструментов
TOOLS = [get_current_time, read_file, list_directory, calculate, word_count]

# ============ СИСТЕМНЫЙ ПРОМПТ ============

SYSTEM_PROMPT = """Ты — закупочный AI-ассистент, работающий в командной строке.
Твои возможности:
- Отвечать на вопросы
- Работать с файловой системой (читать файлы, показывать содержимое директорий)
- Вычислять математические выражения
- Показывать текущее время
- Анализировать текст (подсчёт слов, символов)

Правила:
1. Если вопрос простой — отвечай сразу, без использования инструментов
2. Если нужна информация из файла или вычисления — используй соответствующий инструмент
3. Отвечай на том же языке, на котором задан вопрос
4. Будь полезным, точным и немногословным
5. Если не уверен — скажи об этом"""

# ============ НАСТРОЙКА МОДЕЛИ ============

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

# ============ СОЗДАНИЕ АГЕНТА ============

def create_agent():
    """Создание агента с инструментами"""
    llm = create_llm()
    
    # Создаём промпт с системным сообщением и плейсхолдерами
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Создаём агента с поддержкой tool calling
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    
    # Оборачиваем в executor с настройками
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,  # True — показывать логи вызовов инструментов
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,  # Автоматически обрабатывать ошибки парсинга
        return_intermediate_steps=False,  # Не возвращать шаги в финальный ответ
    )
    
    return agent_executor

# ============ CLI ИНТЕРФЕЙС ============

def print_welcome():
    """Вывод приветственного сообщения"""
    welcome_text = """
# 🤖 AI Agent CLI
Модель: {MODEL_NAME} (vLLM)
Инструменты: время, файлы, калькулятор, текст

Команды:
  • /help    — показать справку
  • /clear   — очистить историю диалога
  • /tools   — показать доступные инструменты
  • /exit    — выход
"""
    console.print(Panel(Markdown(welcome_text), border_style="cyan"))

def print_tools():
    """Вывод списка инструментов"""
    tools_info = "## 🔧 Доступные инструменты\n\n"
    for tool in TOOLS:
        tools_info += f"**{tool.name}**: {tool.description.split('.')[0]}\n"
    console.print(Panel(Markdown(tools_info), border_style="green"))

def process_user_input(agent_executor: AgentExecutor, user_input: str, 
                       chat_history: List) -> tuple[str, List]:
    """
    Обработка пользовательского ввода через агента
    
    Returns:
        tuple: (ответ агента, обновлённая история)
    """
    try:
        # Показываем индикатор выполнения
        with console.status("[cyan]Думаю...[/cyan]", spinner="dots"):
            # Вызов агента
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        
        output = result["output"]
        
        # Обновляем историю диалога
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        return output, chat_history
        
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        return error_msg, chat_history

def main():
    """Главный цикл программы"""
    # Проверка соединения с vLLM
    console.print("[yellow]Проверка соединения с vLLM...[/yellow]")
    try:
        llm = create_llm()
        test_response = llm.invoke("ping")
        console.print(f"[green]✓ Соединение установлено. Модель: {MODEL_NAME}[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Ошибка подключения: {e}[/red]")
        console.print("[red]Убедитесь, что vLLM запущен[/red]")
        sys.exit(1)
    
    # Создаём агента
    agent_executor = create_agent()
    chat_history = []
    
    print_welcome()
    
    # Главный цикл
    while True:
        try:
            # Запрос ввода
            user_input = Prompt.ask("\n[bold cyan]Вы[/bold cyan]")
            
            # Обработка служебных команд
            if user_input.strip().lower() in ['/exit', '/quit', '/q']:
                console.print("[yellow]До свидания![/yellow]")
                break
            
            elif user_input.strip().lower() == '/help':
                console.print(Panel(
                    "Служебные команды:\n"
                    "  /help  — эта справка\n"
                    "  /clear — очистить историю\n"
                    "  /tools — список инструментов\n"
                    "  /exit  — выход\n\n"
                    "Просто введите ваш вопрос или задачу!",
                    title="Справка",
                    border_style="blue"
                ))
                continue
            
            elif user_input.strip().lower() == '/clear':
                chat_history.clear()
                console.print("[green]✓ История диалога очищена[/green]")
                continue
            
            elif user_input.strip().lower() == '/tools':
                print_tools()
                continue
            
            # Пропускаем пустой ввод
            if not user_input.strip():
                continue
            
            # Обработка запроса
            response, chat_history = process_user_input(
                agent_executor, user_input, chat_history
            )
            
            # Вывод ответа с поддержкой Markdown
            console.print("\n[bold green]🤖 Агент[/bold green]")
            console.print(Panel(Markdown(response), border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Прерывание. Для выхода введите /exit[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]До свидания![/yellow]")
            break

if __name__ == "__main__":
    main()
