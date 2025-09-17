# agent.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Providers
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Tools
from tools import (
    web_search, wiki_summary, wiki_page, youtube_transcript,
    fetch_attachment, transcribe_audio, run_python_file, excel_food_sales_total,
    botany_vegetables_from_list, fetch_url_text
)
from langchain_core.tools import tool

load_dotenv()

# -------------------------------
# Simple math tools (docstrings required by @tool)
# -------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the product."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a and return the difference."""
    return a - b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b and return the quotient. Raises an error if b == 0."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Return the remainder when a is divided by b."""
    return a % b

@tool
def parse_date_range(date_range_text: str) -> str:
    """Parse a date range text and return structured information about start, end, and inclusivity.
    Input: text containing date range like '2000 to 2009, inclusive' or 'between 1995 and 2000'
    Returns: structured information about the date range requirements."""
    import re
    
    text = date_range_text.lower().strip()
    
    # Extract years
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    
    # Check for inclusive/exclusive
    is_inclusive = 'inclusive' in text
    is_exclusive = 'exclusive' in text
    
    # Check for range indicators
    has_to = ' to ' in text
    has_between = 'between' in text
    has_from = 'from' in text
    
    if len(years) >= 2:
        start_year = int(years[0])
        end_year = int(years[1])
        
        result = f"Date range: {start_year} to {end_year}"
        if is_inclusive:
            result += " (INCLUSIVE - includes both start and end years)"
        elif is_exclusive:
            result += " (EXCLUSIVE - excludes both start and end years)"
        else:
            result += " (default: inclusive)"
        
        return result
    else:
        return f"Could not parse date range from: {date_range_text}"

# -------------------------------
# System prompt
# -------------------------------
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_text = f.read()
sys_msg = SystemMessage(content=system_prompt_text)

# -------------------------------
# Tools set
# -------------------------------
TOOLS = [
    multiply, add, subtract, divide, modulus, parse_date_range,
    web_search, wiki_summary, wiki_page, youtube_transcript,
    fetch_attachment, transcribe_audio, run_python_file, excel_food_sales_total,
    botany_vegetables_from_list, fetch_url_text
]

# -------------------------------
# Settings (env-configurable)
# -------------------------------
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "15"))
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "3"))

# -------------------------------
# Build graph
# -------------------------------
def build_graph(provider: str = DEFAULT_PROVIDER):
    # LLM provider select
    if provider == "openai":
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
    elif provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-7B-Instruct",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.2,
            ),
        )
    else:
        raise ValueError("Invalid PROVIDER. Use: openai (default), google, groq, huggingface.")

    llm_with_tools = llm.bind_tools(TOOLS)

    def retriever_node(state: MessagesState):
        # Prepend system message at start
        messages = [sys_msg] + state.get("messages", [])
        
        # Add a specific instruction for detailed analysis if the question contains specific criteria
        if state.get("messages"):
            last_message = state["messages"][-1]
            if last_message and isinstance(last_message, HumanMessage):
                content = last_message.content.lower()
                # Check for specific indicators that require careful attention
                specific_indicators = [
                    "inclusive", "exclusive", "from", "to", "between", "during",
                    "exactly", "precisely", "only", "strictly", "specifically"
                ]
                if any(indicator in content for indicator in specific_indicators):
                    # Add a reminder message about specificity
                    reminder = SystemMessage(content="REMINDER: This question contains specific criteria. Pay extra attention to exact requirements like date ranges, inclusive/exclusive terms, and precise specifications.")
                    messages.append(reminder)
        
        return {"messages": messages}

    def assistant(state: MessagesState):
        try:
            messages = state.get("messages", [])
            if not messages:
                return {"messages": []}
            return {"messages": [llm_with_tools.invoke(messages)]}
        except Exception as e:
            # Return an error message if the LLM fails
            error_msg = AIMessage(content=f"Error in assistant: {str(e)}")
            return {"messages": [error_msg]}

    def should_continue(state: MessagesState) -> str:
        # Check if messages list is empty
        if not state["messages"]:
            return "__end__"
            
        last_message = state["messages"][-1]
        
        # Count tool calls in the conversation
        tool_call_count = 0
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tool_call_count += len(msg.tool_calls)
        
        # If we've made too many tool calls, stop
        if tool_call_count >= MAX_TOOL_CALLS:
            return "__end__"
        
        # If the LLM makes a tool call, then we route to the "tools" node
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, we stop (reply to the user)
        return "__end__"

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever_node)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", should_continue)
    builder.add_edge("tools", "assistant")

    return builder.compile()

class LangGraphAgent:
    def __init__(self, provider: str = DEFAULT_PROVIDER, recursion_limit: int = RECURSION_LIMIT):
        self.graph = build_graph(provider)
        self.recursion_limit = recursion_limit

    def run_with_trace(self, question: str) -> Dict[str, Any]:
        try:
            # Start with a clean conversation to avoid tool call corruption
            clean_messages = [HumanMessage(content=question)]
            
            # Invoke the graph with recursion limit
            result = self.graph.invoke(
                {"messages": clean_messages},
                config={"recursion_limit": self.recursion_limit}
            )
            msgs: List = result["messages"]

            # Build trace
            thoughts, actions = [], []
            if msgs:  # Check if msgs is not empty
                for m in msgs:
                    try:
                        if isinstance(m, AIMessage):
                            if isinstance(m.content, str) and m.content.strip():
                                thoughts.append(m.content.strip()[:100])
                            if getattr(m, "tool_calls", None):
                                for tc in m.tool_calls:
                                    actions.append(f"{tc['name']}")
                        elif isinstance(m, ToolMessage):
                            # Add tool results to observations
                            if hasattr(m, 'content') and m.content:
                                thoughts.append(f"Tool result: {str(m.content)[:100]}")
                    except Exception as e:
                        # Skip problematic messages
                        continue
            
            trace = {"thoughts": thoughts[:3], "actions": actions[:3], "observations": []}
            
            # Get the final answer - look for the last AI message that's not a tool call
            final_text = ""
            if msgs:  # Check if msgs is not empty
                try:
                    for m in reversed(msgs):
                        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                            final_text = m.content.strip()
                            break
                    
                    # If no final answer found, use the last message
                    if not final_text and msgs:
                        last_msg = msgs[-1]
                        if hasattr(last_msg, 'content'):
                            final_text = str(last_msg.content)
                        else:
                            final_text = str(last_msg)
                except Exception as e:
                    # If there's an error in message processing, use a default
                    final_text = "Error in message processing"
            
            # Clean up the answer - remove tool names and intermediate steps
            if final_text:
                lines = final_text.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip lines that are tool names or intermediate steps
                    if (line and 
                        not line.startswith(('fetch_attachment:', 'web_search:', 'transcribe_audio:', 'excel_food_sales_total:', 'wiki_')) and
                        not line.startswith(('I will', 'Let me', 'I need to', 'I should', 'I can', 'I\'ll')) and
                        not line.startswith(('First,', 'Next,', 'Then,', 'Finally,', 'Now I', 'I\'m going to')) and
                        not line.startswith(('Based on', 'According to', 'The search results', 'The data shows'))):
                        clean_lines.append(line)
                
                if clean_lines:
                    # Look for the most specific answer (longest line that seems like a final answer)
                    best_answer = clean_lines[-1]  # Default to last line
                    for line in clean_lines:
                        if len(line) > len(best_answer) and not line.startswith(('The', 'This', 'It')):
                            best_answer = line
                    final_text = best_answer
            
            # Ensure FINAL ANSWER format
            if not final_text.strip().startswith("FINAL ANSWER:"):
                final_text = f"FINAL ANSWER: {final_text.strip()}"
            
            return {"final": final_text, "trace": trace}
            
        except Exception as e:
            # Return error with more specific information
            error_msg = f"FINAL ANSWER: Error processing question: {str(e)}"
            if "list index out of range" in str(e):
                error_msg = "FINAL ANSWER: Error processing question: Message processing error - please try rephrasing your question."
            return {"final": error_msg, "trace": {"thoughts": [], "actions": [], "observations": []}}
