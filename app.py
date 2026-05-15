import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "google-genai"], check=True)

import os
import json
import time
import datetime
import gradio as gr
import google.genai as genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY             = os.environ["GEMINI_API_KEY"]
STORE_NAME          = os.environ.get("STORE_NAME", "fileSearchStores/userstudystore-3gytybx82f4t")
RESEARCHER_PASSWORD = os.environ.get("RESEARCHER_PASSWORD", "mypassword123")
MODEL               = "gemini-2.5-flash"
LOG_FILE            = "experiment_log.jsonl"
CSV_PATH            = "small_dataset.csv"

client = genai.Client(api_key=API_KEY)

file_search_tool = types.Tool(
    file_search=types.FileSearch(file_search_store_names=[STORE_NAME])
)

# ── Experiment Tasks ──────────────────────────────────────────────────────────
TASKS = [
    {
        "id": 1,
        "title": "Find a ritual object",
        "description": "Find any ritual object in the collection. What is it called, when was it made, and what culture does it come from?",
        "hint": "Try searching for 'ritual object' or 'ritueel'",
    },
    {
        "id": 2,
        "title": "Identify the oldest item",
        "description": "What is the oldest item in the dataset? Provide its name, ID, and creation date.",
        "hint": "Think about negative creation dates (B.C.)",
    },
    {
        "id": 3,
        "title": "Find lacquerware",
        "description": "Find two examples of lacquerware in the collection. Who made them and when?",
        "hint": "Try 'lakwerk' or 'lacquer'",
    },
    {
        "id": 4,
        "title": "Artworks from 1700",
        "description": "How many items in the collection were created around 1700? List at least three.",
        "hint": "Search for items by creation date",
    },
    {
        "id": 5,
        "title": "Anonymous creators",
        "description": "Find three items created by anonymous makers. What types of objects are they?",
        "hint": "Many items have 'anonymous' as creator",
    },
]

# ── Logging ───────────────────────────────────────────────────────────────────
def log_event(participant_id, condition, event_type, data):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "participant_id": participant_id,
        "condition": condition,
        "event": event_type,
    }
    entry.update(data)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Condition A: Simple keyword search ────────────────────────────────────────
def keyword_search(query):
    import pandas as pd
    results = []
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="replace")
        q = query.lower()
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
        matched = df[mask].head(8)

        for _, row in matched.iterrows():
            results.append({
                "id": str(row.get("ID", "")),
                "title": str(row.get("Title", "Unknown")),
                "type": str(row.get("Type", "")),
                "creator": str(row.get("Creator", "Anonymous")),
                "date": str(row.get("Creation_Date", "")),
                "url": str(row.get("Identifier", "")),
                "image": str(row.get("Image_URL", "")),
            })
    except Exception as e:
        results = [{"title": f"Error: {e}", "id": "", "type": "", "creator": "", "date": "", "url": "", "image": ""}]
    return results

def search_condition_a(query, state):
    if not query.strip():
        return "<p style='color:#888'>Enter a search term above.</p>", state

    t_start = time.time()
    results = keyword_search(query)
    elapsed = round(time.time() - t_start, 3)

    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"] = state.get("queries", []) + [query]

    log_event(state.get("participant_id", "unknown"), "A", "search", {
        "query": query,
        "query_length": len(query),
        "query_number": state.get("query_count", 0),
        "result_count": len(results),
        "elapsed_s": elapsed,
    })

    return f"{len(results)} results", state

# ── Condition B: AI Chat (FIXED FOR YOUR GRADIO VERSION) ─────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum research assistant. Answer using sources when possible. "
    "If unsure, say so explicitly."
)

def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"] = state.get("queries", []) + [message]

    t_start = time.time()

    # ── FIX: convert old tuple history to Gemini format ──
    contents = []
    for turn in (history or []):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, assistant_msg = turn

            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=str(user_msg))]
            ))
            contents.append(types.Content(
                role="model",
                parts=[types.Part(text=str(assistant_msg))]
            ))

    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=message)]
    ))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_B,
                tools=[file_search_tool],
            )
        )

        answer = response.text or "No response generated."

    except Exception as e:
        answer = f"Error: {e}"

    elapsed = round(time.time() - t_start, 3)

    log_event(state.get("participant_id", "unknown"), "B", "chat", {
        "query": message,
        "response_text": answer,
        "response_length": len(answer),
        "elapsed_s": elapsed,
        "query_number": state.get("query_count", 0),
    })

    # ── FIX: KEEP OLD GRADIO FORMAT ──
    new_history = (history or []) + [[message, answer]]

    return "", new_history, state, new_history

# ── Survey submission ─────────────────────────────────────────────────────────
def submit_survey(pid, condition, task_id, task_completed, completion_time,
                  q_trust, q_useful, q_ease, q_accurate, q_recommend,
                  q_toast_reliable, q_toast_confident, q_toast_trustworthy,
                  q_verified, q_comments, state):

    survey_data = {
        "participant_id": pid,
        "condition": condition,
        "task_id": task_id,
        "task_completed": task_completed,
        "self_reported_time_min": completion_time,
        "trust": q_trust,
        "usefulness": q_useful,
        "ease_of_use": q_ease,
        "accuracy": q_accurate,
        "recommend": q_recommend,
        "toast_reliable": q_toast_reliable,
        "toast_confident": q_toast_confident,
        "toast_trustworthy": q_toast_trustworthy,
        "verified_sources": q_verified,
        "comments": q_comments,
        "query_count": state.get("query_count", 0),
        "queries": state.get("queries", []),
    }

    log_event(pid, condition, "survey", survey_data)

    return (
        f"Survey submitted! Thank you, {pid}.\n\n"
        f"Condition: {condition} · Task: {task_id} · Queries: {state.get('query_count', 0)}"
    )

# ── Session end ───────────────────────────────────────────────────────────────
def end_session(state):
    elapsed = round(time.time() - state.get("session_start", time.time()), 1)

    log_event(state.get("participant_id", "unknown"), state.get("condition", "?"), "session_end", {
        "task_id": state.get("task_id"),
        "total_time_s": elapsed,
        "total_queries": state.get("query_count", 0),
        "all_queries": state.get("queries", []),
    })

    return f"Session ended. Time: {elapsed}s · Queries: {state.get('query_count', 0)}"

# ── Download / logs ───────────────────────────────────────────────────────────
def download_log():
    return LOG_FILE if os.path.exists(LOG_FILE) else None

def load_log():
    try:
        with open(LOG_FILE) as f:
            return f.read()
    except:
        return "No logs yet."

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Museum Collection Experiment") as demo:

    session_state = gr.State({})
    chat_history = gr.State([])

    gr.Markdown("# Museum Study")

    # ── Setup ──
    with gr.Tab("Setup"):
        pid = gr.Textbox(label="Participant ID")
        cond = gr.Dropdown(["A — Keyword Search", "B — AI Chat"])
        task = gr.Dropdown([f"Task {t['id']}: {t['title']}" for t in TASKS])

        out = gr.Markdown()
        btn = gr.Button("Start Session")

        def start_session(pid, cond, task_label, state):
            task_id = int(task_label.split(":")[0].replace("Task ", ""))
            task_obj = next(t for t in TASKS if t["id"] == task_id)

            state["participant_id"] = pid
            state["condition"] = cond
            state["task_id"] = task_id
            state["session_start"] = time.time()
            state["query_count"] = 0
            state["queries"] = []

            log_event(pid, cond, "session_start", {"task_id": task_id})

            return (
                f"Task: {task_obj['description']}\nHint: {task_obj['hint']}",
                state
            )

        btn.click(start_session, [pid, cond, task, session_state], [out, session_state])

    # ── Keyword ──
    with gr.Tab("Keyword Search"):
        q = gr.Textbox()
        btn2 = gr.Button("Search")
        out2 = gr.Textbox()

        btn2.click(search_condition_a, [q, session_state], [out2, session_state])

    # ── Chat (FIXED ONLY HERE) ──
    with gr.Tab("AI Chat"):

        chatbot = gr.Chatbot(height=450)  # IMPORTANT FIX

        msg = gr.Textbox()
        send = gr.Button("Send")

        send.click(
            chat_condition_b,
            [msg, chat_history, session_state],
            [msg, chatbot, session_state, chat_history]
        )

        msg.submit(
            chat_condition_b,
            [msg, chat_history, session_state],
            [msg, chatbot, session_state, chat_history]
        )

demo.launch()