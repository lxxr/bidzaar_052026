# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:11:48 2026

@author: rublev.an
"""
#!/usr/bin/env python3
"""
CLI AI Agent на базе LangChain + vLLM + MCP Tools
Использует асинхронный AgentExecutor для поддержки MCP-инструментов
Запуск: python agent_mcp.py
"""


import sys
import os
import asyncio
from typing import List, Optional, Dict
from contextlib import AsyncExitStack
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from datetime import datetime

# ============ КОНФИГУРАЦИЯ ============

VLLM_BASE_URL = "<your-LLM-base-url>"
VLLM_API_KEY = "<your-api-secret-key>"
MODEL_NAME = "<your-LLM-model-name>"

TEMPERATURE = 0.0
MAX_ITERATIONS = 10

# ============ КОНФИГУРАЦИЯ MCP-СЕРВЕРОВ ============

MCP_STDIO_SERVERS = {
    "bidzaar": {
        "command": "python",
        "args": ["/path-to/bidzaar_mcp.py"],
        "transport": "stdio",
    },
    "ddgs": {
        "command": "ddgs",
        "args": ["mcp"],
        "transport": "stdio",
    },
}

# ============ ИНИЦИАЛИЗАЦИЯ ============

console = Console()

# ============ ЛОКАЛЬНЫЕ ИНСТРУМЕНТЫ ============

@tool
def get_current_time() -> str:
    """Получить текущее время и дату."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def read_file(filepath: str) -> str:
    """Прочитать содержимое файла."""
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
    """Показать содержимое директории."""
    try:
        files = os.listdir(path)
        if not files:
            return f"Директория {path} пуста"
        return f"Содержимое {path}:\n" + "\n".join(f"  • {f}" for f in sorted(files))
    except Exception as e:
        return f"Ошибка: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """Вычислить математическое выражение."""
    try:
        allowed_chars = set("0123456789+-*/().%** ")
        if not all(c in allowed_chars for c in expression):
            return "Ошибка: выражение содержит недопустимые символы"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Результат: {expression} = {result}"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"

@tool
def word_count(text: str) -> str:
    """Подсчитать количество слов и символов в тексте."""
    words = text.split()
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return f"Слов: {len(words)}, Символов (всего): {chars}, Символов (без пробелов): {chars_no_spaces}"

LOCAL_TOOLS = [get_current_time, read_file, list_directory, calculate, word_count]

# ============ СИСТЕМНЫЙ ПРОМПТ ============

SYSTEM_PROMPT = """Ты — закупочный AI-ассистент, работающий в командной строке.
Твои возможности:
- Отвечать на вопросы
- Работать с файловой системой (читать файлы, показывать содержимое директорий)
- Вычислять математические выражения
- Показывать текущее время
- Анализировать текст (подсчёт слов, символов)
- Искать информацию в интернете (через ddgs)
- Работать с закупочными данными (через bidzaar)

Правила:
1. Если вопрос простой — отвечай сразу, без использования инструментов
2. Если нужна информация из файла или вычисления — используй соответствующий инструмент
3. Для поиска в интернете используй инструменты ddgs
4. Для работы с закупками используй инструменты bidzaar
5. Отвечай на том же языке, на котором задан вопрос
6. Будь полезным, точным и немногословным
7. Если не уверен — скажи об этом"""

# ============ НАСТРОЙКА МОДЕЛИ ============

def create_llm():
    """Создание LLM-клиента для vLLM"""
    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=VLLM_API_KEY,
        openai_api_base=VLLM_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=4096,
        timeout=120,
        max_retries=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

# ============ ЗАГРУЗКА MCP-ИНСТРУМЕНТОВ С ПОСТОЯННЫМИ СЕССИЯМИ ============

async def load_mcp_tools_persistent() -> tuple[List, AsyncExitStack]:
    """
    Загружает MCP-инструменты и сохраняет соединения открытыми.
    
    Returns:
        tuple: (список инструментов, AsyncExitStack для управления соединениями)
    """
    mcp_tools = []
    exit_stack = AsyncExitStack()
    
    try:
        console.print("[yellow]Подключение к MCP-серверам (постоянные сессии)...[/yellow]")
        
        for server_name, config in MCP_STDIO_SERVERS.items():
            try:
                console.print(f"  Подключение к '{server_name}'...", end=" ")
                
                # Создаём клиент для каждого сервера отдельно
                client = MultiServerMCPClient({server_name: config})
                
                # Открываем сессию и сохраняем её через exit_stack
                session = await exit_stack.enter_async_context(
                    client.session(server_name)
                )
                
                # Загружаем инструменты из открытой сессии
                tools = await load_mcp_tools(session)
                
                if tools:
                    mcp_tools.extend(tools)
                    console.print(f"[green]✓ {len(tools)} инструментов[/green]")
                else:
                    console.print("[yellow]нет инструментов[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]✗ {e}[/red]")
                continue
        
        if mcp_tools:
            console.print(f"\n[green]✓ Всего загружено {len(mcp_tools)} MCP-инструментов:[/green]")
            for tool in mcp_tools:
                desc = getattr(tool, 'description', 'Нет описания')
                console.print(f"  • [cyan]{tool.name}[/cyan] - {desc[:100]}")
            console.print()
        else:
            console.print("\n[yellow]⚠ MCP-серверы не предоставили инструментов[/yellow]")
    
    except Exception as e:
        console.print(f"\n[red]✗ Ошибка загрузки MCP: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return mcp_tools, exit_stack


# Альтернативный способ: через get_tools (если поддерживается)
async def load_mcp_tools_simple() -> List:
    """
    Простая загрузка через get_tools.
    Может работать, если клиент сохраняет соединения внутри.
    """
    mcp_tools = []
    
    try:
        console.print("[yellow]Подключение к MCP-серверам...[/yellow]")
        
        # Создаём клиент для всех серверов
        client = MultiServerMCPClient(MCP_STDIO_SERVERS)
        
        # Получаем инструменты (клиент должен сам управлять соединениями)
        tools = await client.get_tools()
        mcp_tools.extend(tools)
        
        if mcp_tools:
            console.print(f"[green]✓ Загружено {len(mcp_tools)} MCP-инструментов:[/green]")
            for tool in mcp_tools:
                desc = getattr(tool, 'description', 'Нет описания')
                console.print(f"  • [cyan]{tool.name}[/cyan] - {desc[:100]}")
            console.print()
        else:
            console.print("[yellow]⚠ MCP-серверы не предоставили инструментов[/yellow]")
            
    except Exception as e:
        console.print(f"[red]✗ Ошибка загрузки MCP: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    return mcp_tools

# ============ СОЗДАНИЕ АГЕНТА ============

def create_agent(all_tools: List) -> AgentExecutor:
    """Создание агента с инструментами"""
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        verbose=False,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
        early_stopping_method="generate",
    )
    
    return agent_executor

# ============ ОБРАБОТКА ЗАПРОСОВ ============

async def process_input_async(
    agent_executor: AgentExecutor,
    user_input: str,
    chat_history: List,
    verbose: bool = False
) -> tuple[str, List]:
    """Асинхронная обработка пользовательского ввода"""
    try:
        original_verbose = agent_executor.verbose
        agent_executor.verbose = verbose
        
        with console.status("[cyan]Думаю...[/cyan]", spinner="dots"):
            result = await agent_executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history,
            })
        
        agent_executor.verbose = original_verbose
        output = result["output"]
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        return output, chat_history
        
    except Exception as e:
        agent_executor.verbose = original_verbose
        error_msg = f"❌ Ошибка: {str(e)}"
        import traceback
        console.print(f"[dim red]{traceback.format_exc()}[/dim red]")
        return error_msg, chat_history

# ============ CLI ИНТЕРФЕЙС ============

def print_welcome(mcp_tools_count: int = 0):
    """Вывод приветственного сообщения"""
    total_tools = len(LOCAL_TOOLS) + mcp_tools_count
    welcome_text = f"""
# 🤖 AI Agent CLI (с MCP-инструментами)
Модель: {MODEL_NAME} (vLLM)
Инструменты: {len(LOCAL_TOOLS)} локальных + {mcp_tools_count} MCP = {total_tools} всего

Команды:
  • /help    — показать справку
  • /clear   — очистить историю диалога
  • /tools   — показать доступные инструменты
  • /mcp     — статус MCP-подключений
  • /verbose — переключить отладку
  • /exit    — выход
"""
    console.print(Panel(Markdown(welcome_text), border_style="cyan"))

def print_tools(all_tools: List):
    """Вывод списка всех инструментов"""
    local_names = {t.name for t in LOCAL_TOOLS}
    
    tools_info = "## 🔧 Доступные инструменты\n\n"
    tools_info += "### 📦 Локальные:\n"
    for tool in all_tools:
        if tool.name in local_names:
            desc = getattr(tool, 'description', 'Нет описания')
            tools_info += f"- **{tool.name}**: {desc.split('.')[0]}\n"
    
    tools_info += "\n### 🌐 MCP:\n"
    mcp_count = 0
    for tool in all_tools:
        if tool.name not in local_names:
            desc = getattr(tool, 'description', 'Нет описания')
            tools_info += f"- **{tool.name}**: {desc[:120]}...\n"
            mcp_count += 1
    if mcp_count == 0:
        tools_info += "*(нет — MCP-серверы не подключены)*\n"
    
    console.print(Panel(Markdown(tools_info), border_style="green"))

def print_mcp_status(mcp_tools_count: int):
    """Вывод статуса MCP-подключений"""
    if mcp_tools_count == 0:
        console.print(Panel(
            "[red]MCP-серверы не подключены.[/red]",
            title="🌐 MCP Статус",
            border_style="red"
        ))
        return
    
    status_text = f"[green]Подключено серверов: {len(MCP_STDIO_SERVERS)}[/green]\n"
    status_text += f"[green]Загружено инструментов: {mcp_tools_count}[/green]\n\n"
    status_text += "Серверы:\n"
    for name, cfg in MCP_STDIO_SERVERS.items():
        status_text += f"  • [cyan]{name}[/cyan] → {cfg['command']} {' '.join(cfg['args'])}\n"
    
    console.print(Panel(status_text, title="🌐 MCP Статус", border_style="blue"))

# ============ ГЛАВНАЯ АСИНХРОННАЯ ФУНКЦИЯ ============

async def async_main():
    """Асинхронная главная функция с постоянными MCP-сессиями"""
    
    # Проверка vLLM
    console.print("[yellow]Проверка соединения с vLLM...[/yellow]")
    try:
        llm = create_llm()
        test_response = llm.invoke("ping")
        console.print(f"[green]✓ Соединение установлено. Модель: {MODEL_NAME}[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Ошибка подключения: {e}[/red]")
        return
    
    # Загружаем MCP-инструменты
    mcp_tools, exit_stack = await load_mcp_tools_persistent()
    
    try:
        # Объединяем инструменты
        all_tools = LOCAL_TOOLS + mcp_tools
        
        # Создаём агента
        agent_executor = create_agent(all_tools)
        chat_history = []
        verbose_mode = False
        
        print_welcome(len(mcp_tools))
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]Вы[/bold cyan]")
                
                if user_input.strip().lower() in ['/exit', '/quit', '/q']:
                    console.print("[yellow]До свидания![/yellow]")
                    break
                
                elif user_input.strip().lower() == '/help':
                    console.print(Panel(
                        "Служебные команды:\n"
                        "  /help    — эта справка\n"
                        "  /clear   — очистить историю\n"
                        "  /tools   — список инструментов\n"
                        "  /mcp     — статус MCP\n"
                        "  /verbose — переключить отладку\n"
                        "  /exit    — выход",
                        title="Справка",
                        border_style="blue"
                    ))
                    continue
                
                elif user_input.strip().lower() == '/clear':
                    chat_history.clear()
                    console.print("[green]✓ История очищена[/green]")
                    continue
                
                elif user_input.strip().lower() == '/tools':
                    print_tools(all_tools)
                    continue
                
                elif user_input.strip().lower() == '/mcp':
                    print_mcp_status(len(mcp_tools))
                    continue
                
                elif user_input.strip().lower() == '/verbose':
                    verbose_mode = not verbose_mode
                    status = "включен" if verbose_mode else "выключен"
                    console.print(f"[yellow]Режим отладки {status}[/yellow]")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Асинхронная обработка запроса
                response, chat_history = await process_input_async(
                    agent_executor, user_input, chat_history, verbose_mode
                )
                
                console.print("\n[bold green]🤖 Агент[/bold green]")
                console.print(Panel(Markdown(response), border_style="green"))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Прерывание. Для выхода /exit[/yellow]")
                continue
            except EOFError:
                console.print("\n[yellow]До свидания![/yellow]")
                break
    
    finally:
        console.print("[dim]Закрытие MCP-сессий...[/dim]")
        await exit_stack.aclose()

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
