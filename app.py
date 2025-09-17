# app.py
import os, re
import threading
from dotenv import load_dotenv
import gradio as gr
import requests
import pandas as pd

from agent import LangGraphAgent

# Global stop flag
stop_flag = threading.Event()

# Load environment variables early so HF_TOKEN is available before Gradio initializes
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DOTENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH)

# Normalize HF token environment variable and perform programmatic login for local runs
try:
    from huggingface_hub import login as hf_login
except Exception:  # huggingface_hub is a transitive dep of gradio
    hf_login = None

_token_candidates = [
    os.getenv("HF_TOKEN"),
    os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    os.getenv("HUGGINGFACE_HUB_TOKEN"),
    os.getenv("HUGGINGFACE_TOKEN"),
]
_token = next((t for t in _token_candidates if t), None)
if _token and not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _token
    try:
        if hf_login:
            hf_login(token=_token, add_to_git_credential=False)
    except Exception:
        pass

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def _only_answer_from_final(text: str) -> str:
    if not text: return ""
    m = re.search(r"FINAL ANSWER:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip()
        if ans.startswith("[") and ans.endswith("]"):
            ans = ans[1:-1].strip()
        return ans.splitlines()[-1].strip()
    return text.strip().splitlines()[-1].strip()

def _format_trace_md(task_id: str, trace: dict) -> str:
    md = [f"### Task `{task_id}`"]
    for t in trace.get("thoughts", []):
        md.append(f"- **Thought:** {t}")
    for a in trace.get("actions", []):
        md.append(f"- **Action:** `{a}`")
    return "\n".join(md)

def run_and_submit_all(profile: gr.OAuthProfile | None):
    global stop_flag
    stop_flag.clear()  # Reset stop flag
    
    # Locally you probably won't log in; handle both cases
    space_id = os.getenv("SPACE_ID")
    username = profile.username.strip() if profile else "local_user"

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "local-run"

    # Instantiate agent
    try:
        # Use OpenAI if API key is available, otherwise fallback to Google (free)
        provider = os.getenv("PROVIDER", "openai")  # openai default (now that you have API key)
        agent = LangGraphAgent(provider=provider)
    except Exception as e:
        yield f"Error initializing agent: {e}", None, ""
        return

    # Fetch questions
    yield "Fetching questions...", None, ""
    try:
        r = requests.get(questions_url, timeout=30)
        r.raise_for_status()
        questions = r.json()
        if not isinstance(questions, list) or not questions:
            yield "Fetched questions list is empty or invalid.", None, ""
            return
    except Exception as e:
        yield f"Error fetching questions: {e}", None, ""
        return

    # Run agent streamed
    results_log, answers_payload, live_blocks = [], [], []
    N = len(questions)
    for i, item in enumerate(questions, 1):
        if stop_flag.is_set():
            yield "Stopped by user", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)
            return
            
        task_id = item.get("task_id")
        qtext = item.get("question", "")
        file_name = (item.get("file_name") or "").strip()
        if file_name:
            file_url = f"{api_url}/files/{file_name}"
            qtext = f"(If needed, fetch attachment here: {file_url})\n" + qtext
        try:
            out = agent.run_with_trace(qtext)
            final_raw = out["final"]
            final_ans = _only_answer_from_final(final_raw)
            answers_payload.append({"task_id": task_id, "submitted_answer": final_ans})
            results_log.append({"Task ID": task_id, "Question": qtext, "Submitted Answer": final_ans})
            live_blocks.append(_format_trace_md(task_id, out["trace"]))
            yield f"[{i}/{N}] Answered {task_id}", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": qtext, "Submitted Answer": f"AGENT ERROR: {e}"})
            yield f"[{i}/{N}] ERROR on {task_id}: {e}", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)

    if not answers_payload:
        yield "Agent did not produce any answers to submit.", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)
        return

    # Submit
    submission = {"username": username, "agent_code": agent_code, "answers": answers_payload}
    yield f"Submitting {len(answers_payload)} answers for '{username}'...", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)

    try:
        r = requests.post(submit_url, json=submission, timeout=120)
        r.raise_for_status()
        data = r.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {data.get('username')}\n"
            f"Overall Score: {data.get('score','N/A')}% "
            f"({data.get('correct_count','?')}/{data.get('total_attempted','?')} correct)\n"
            f"Message: {data.get('message','No message.')}"
        )
        yield final_status, pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text[:500]
        yield f"Submission Failed: {e} | {detail}", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)
    except Exception as e:
        yield f"Submission Failed: {e}", pd.DataFrame(results_log), "\n\n---\n\n".join(live_blocks)

def stop_evaluation():
    """Stop the evaluation process."""
    global stop_flag
    stop_flag.set()
    return "Stopping evaluation...", None, ""


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧪 LangGraph Agent — Production Runner")
    gr.Markdown("Production-ready agent that uses tools properly and returns definitive answers.")
    gr.LoginButton()
    
    with gr.Row():
        run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
        stop_button = gr.Button("Stop", variant="stop")
    
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)
    live_log = gr.Markdown(label="Agent Trace (live)")
    
    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table, live_log])
    stop_button.click(fn=stop_evaluation, outputs=[status_output, results_table, live_log])

if __name__ == "__main__":
    print("\n----- Production App Starting -----")
    demo.launch(debug=False, share=False)
