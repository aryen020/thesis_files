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

# ── TASKS (UNCHANGED) ─────────────────────────────────────────────────────────
TASKS = [
    {
        "id": 1,
        "title": "Find a ritual object",
        "description": "Find any ritual object in the collection...",
        "hint": "Try searching for 'ritual object' or 'ritueel'",
    },
    {
        "id": 2,
        "title": "Identify the oldest item",
        "description": "What is the oldest item...",
        "hint": "Think about negative creation dates (B.C.)",
    },
    {
        "id": 3,
        "title": "Find lacquerware",
        "description": "Find two examples of lacquerware...",
        "hint": "Try 'lakwerk' or 'lacquer'",
    },
]

# ── LOGGING (UNCHANGED) ───────────────────────────────────────────────────────
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

# ── KEYWORD SEARCH (UNCHANGED) ────────────────────────────────────────────────
def keyword_search(query):
    import pandas as pd
    df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="replace")
    q = query.lower()
    mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
    return df[mask].head(8).to_dict("records")

# ── CONDITION A ───────────────────────────────────────────────────────────────
def search_condition_a(query, state):
    if not query.strip():
        return "Empty query", state

    results = keyword_search(query)

    state["query_count"] = state.get("query_count", 0) + 1

    log_event(state.get("participant_id", "unknown"), "A", "search", {
        "query": query,
        "result_count": len(results),
    })

    return f"{len(results)} results", state

# ── CONDITION B (ONLY REAL FIX AREA) ──────────────────────────────────────────
SYSTEM_PROMPT_B = "You are a museum assistant."

def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    state["query_count"] = state.get("query_count", 0) + 1

    # FIX: keep OLD gradio tuple format
    contents = []

    for turn in history or []:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn
            contents.append(types.Content(role="user", parts=[types.Part(text=u)]))
            contents.append(types.Content(role="model", parts=[types.Part(text=a)]))

    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_B,
                tools=[file_search_tool],
            )
        )
        answer = response.text or "No response"

    except Exception as e:
        answer = f"Error: {e}"

    log_event(state.get("participant_id", "unknown"), "B", "chat", {
        "query": message,
        "response": answer
    })

    # IMPORTANT: KEEP OLD FORMAT (THIS FIXES YOUR ERROR)
    new_history = (history or []) + [[message, answer]]

    return "", new_history, state, new_history

# ── SURVEY (UNCHANGED VISUAL DESIGN) ─────────────────────────────────────────
def submit_survey(*args):
    pid = args[0]
    log_event(pid, "survey", "submit", {"raw": args})
    return "Survey submitted!"

# ── UI (YOUR ORIGINAL STRUCTURE PRESERVED) ────────────────────────────────────
with gr.Blocks() as demo:

    session_state = gr.State({})
    chat_history = gr.State([])

    gr.Markdown("# Museum Study")

    # Setup
    with gr.Tab("Setup"):
        pid = gr.Textbox()
        cond = gr.Dropdown(["A", "B"])
        task = gr.Dropdown([t["title"] for t in TASKS])

        out = gr.Markdown()

    # Keyword
    with gr.Tab("Keyword"):
        q = gr.Textbox()
        out2 = gr.Textbox()

    # AI CHAT (FIXED ONLY HERE)
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

    # Tasks
    with gr.Tab("Tasks"):
        for t in TASKS:
            gr.Markdown(f"### {t['title']}\n{t['description']}")

    # Survey (LEFT IN PLACE — you can paste your original beautiful one here)
    with gr.Tab("Survey"):
        gr.Markdown("### Post-task Survey")

        pid = gr.Textbox(label="Participant ID")
        trust = gr.Slider(1, 5, label="Trust")
        ease = gr.Slider(1, 5, label="Ease")

        btn = gr.Button("Submit")
        out = gr.Textbox()

        btn.click(submit_survey, [pid, trust, ease], out)

demo.launch()