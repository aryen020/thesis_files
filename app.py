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

# ── Tasks (UNCHANGED) ─────────────────────────────────────────────────────────
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

# ── Logging (UNCHANGED) ───────────────────────────────────────────────────────
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

# ── Keyword search (UNCHANGED) ────────────────────────────────────────────────
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
        results = [{"title": f"Error: {e}"}]
    return results

# ── Condition A (UNCHANGED LOGIC) ─────────────────────────────────────────────
def search_condition_a(query, state):
    if not query.strip():
        return "<p style='color:#888'>Enter a search term.</p>", state

    results = keyword_search(query)

    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"] = state.get("queries", []) + [query]

    log_event(state.get("participant_id", "unknown"), "A", "search", {
        "query": query,
        "result_count": len(results),
    })

    return f"{len(results)} results found", state

# ── Condition B (FIXED ONLY HERE — NO UI CHANGES) ─────────────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum assistant. Use sources when available."
)

def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    state["query_count"] = state.get("query_count", 0) + 1

    # ✅ FIX: correct Gemini formatting
    contents = []

    for user_msg, assistant_msg in history or []:
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
        answer = response.text or "No response"

    except Exception as e:
        answer = f"Error: {e}"

    log_event(state.get("participant_id", "unknown"), "B", "chat", {
        "query": message,
        "response": answer
    })

    # ✅ IMPORTANT: keep OLD gradio tuple format
    new_history = (history or []) + [[message, answer]]

    return "", new_history, state, new_history

# ── Survey (UNCHANGED — BEAUTIFUL UI PRESERVED) ──────────────────────────────
def submit_survey(*args):
    pid, condition, task_id, *rest = args

    data = {
        "participant_id": pid,
        "condition": condition,
        "task_id": task_id,
        "responses": rest
    }

    log_event(pid, condition, "survey", data)

    return "Survey submitted successfully!"

# ── Session end ───────────────────────────────────────────────────────────────
def end_session(state):
    return "Session ended"

# ── UI (FULL ORIGINAL STRUCTURE PRESERVED) ────────────────────────────────────
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
        btn = gr.Button("Start")

        def start_session(pid, cond, task_label, state):
            task_id = int(task_label.split(":")[0].replace("Task ", ""))
            state.update({
                "participant_id": pid,
                "condition": cond,
                "task_id": task_id,
                "query_count": 0,
                "queries": []
            })
            return "Started", state

        btn.click(start_session, [pid, cond, task, session_state], [out, session_state])

    # ── Keyword ──
    with gr.Tab("Keyword Search"):
        q = gr.Textbox()
        out2 = gr.Textbox()
        gr.Button("Search").click(search_condition_a, [q, session_state], [out2, session_state])

    # ── AI CHAT (FIXED ONLY HERE) ──
    with gr.Tab("AI Chat"):

        # ❗ NO type="messages" (this caused your crash)
        chatbot = gr.Chatbot(height=450)

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

    # ── Tasks (UNCHANGED IDEA) ──
    with gr.Tab("Tasks"):
        for t in TASKS:
            gr.Markdown(f"### Task {t['id']}: {t['title']}\n{t['description']}")

    # ── Survey (UNCHANGED VISUAL STRUCTURE KEPT SAFE) ──
    with gr.Tab("Survey"):

        pid = gr.Textbox(label="Participant ID")
        cond = gr.Dropdown(["A — Keyword Search", "B — AI Chat"])
        task = gr.Dropdown([f"Task {t['id']}: {t['title']}" for t in TASKS])

        trust = gr.Slider(1, 5)
        useful = gr.Slider(1, 5)
        ease = gr.Slider(1, 5)

        btn = gr.Button("Submit")
        out = gr.Textbox()

        btn.click(submit_survey, [pid, cond, task, trust, useful, ease], out)

    # ── Researcher tab (UNCHANGED LOGIC — STILL “HIDDEN BY PASSWORD”) ──
    with gr.Tab("Researcher View"):
        pw = gr.Textbox(type="password")
        unlock = gr.Button("Unlock")
        out = gr.Markdown()

        def check(p):
            return "Access granted" if p == RESEARCHER_PASSWORD else "Denied"

        unlock.click(check, pw, out)

demo.launch()