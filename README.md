---
title: LangGraph Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.37.2"
app_file: app.py
pinned: false
hf_oauth: true
---

# LangGraph Agent Runner - Production Ready

This is a production-ready LangGraph-based agent that uses tools properly and returns definitive answers in the required format.

## Features

- **Production Ready**: Optimized for HF Spaces deployment with proper error handling
- **Multi-tool Agent**: Uses web search, Wikipedia, YouTube transcripts, file processing, and more
- **Definitive Answers**: Returns actual answers, not tool names or intermediate steps
- **Stop Button**: Can stop evaluation mid-process
- **Live Trace**: Shows the agent's thought process and tool usage
- **OAuth Integration**: Works with Hugging Face authentication
- **Multiple LLM Providers**: Supports OpenAI, Google Gemini, Groq, and Hugging Face models

## How to Use

1. **Login** using the Hugging Face login button
2. **Click "Run Evaluation & Submit All Answers"** to start the agent
3. **Use "Stop" button** if you need to halt the process
4. **Watch the live trace** as the agent processes questions
5. **View results** in the table and submission status

## Environment Variables

Set these in your HF Space settings:

- `OPENAI_API_KEY`: Your OpenAI API key (default provider)
- `PROVIDER`: Choose from "openai", "google", "groq", "huggingface"
- `RECURSION_LIMIT`: Maximum tool calls (default: 60)
- `MAX_TOOL_CALLS`: Maximum tool calls per question (default: 8)

## Tools Available

- **Web Search**: DuckDuckGo search
- **Wikipedia**: Article summaries and page content
- **YouTube**: Video transcripts
- **File Processing**: Excel files, attachments
- **Math Tools**: Basic arithmetic operations
- **Text Processing**: URL fetching, text analysis

## Answer Format

The agent strictly follows this format:
```
FINAL ANSWER: [actual answer]
```

No reasoning or explanations are included in the final response.

## Files Included

- `app.py`: Main Gradio application
- `agent.py`: LangGraph agent implementation
- `tools.py`: Tool definitions (web search, Wikipedia, etc.)
- `system_prompt.txt`: System prompt for proper answer formatting
- `requirements.txt`: Python dependencies
- `README.md`: This file
