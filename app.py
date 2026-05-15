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

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY    = os.environ["GEMINI_API_KEY"]
STORE_NAME = os.environ.get("STORE_NAME", "fileSearchStores/userstudystore")
MODEL      = "gemini-2.5-flash"
LOG_FILE   = "experiment_log.jsonl"

client = genai.Client(api_key=API_KEY)

file_search_tool = types.Tool(
    file_search=types.FileSearch(file_search_store_names=[STORE_NAME])
)

# ── TASKS ─────────────────────────────────────────────────────────────────────
TASKS = [
    {"id": 1, "title": "Ritual object", "hint": "ritual object / ritueel"},
    {"id": 2, "title": "Oldest item", "hint": "negative dates / B.C."},
    {"id": 3, "title": "Lacquerware", "hint": "lakwerk / lacquer"},
    {"id": 4, "title": "1700 items", "hint": "1700 / 18th century"},
    {"id": 5, "title": "Anonymous items", "hint": "anonymous"},
]

# ── GROUND TRUTH (H2) ─────────────────────────────────────────────────────────
GROUND_TRUTH = {
    1: ["ritual", "object"],
    2: ["oldest", "bc", "b.c", "date"],
    3: ["lacquer", "lakwerk"],
    4: ["1700", "18th"],
    5: ["anonymous"]
}

def grade_answer(task_id, text):
    if not text:
        return 0.0
    keys = GROUND_TRUTH.get(task_id, [])
    t = text.lower()
    score = sum(1 for k in keys if k in t)
    return round(score / max(len(keys), 1), 2)

# ── LOGGING ───────────────────────────────────────────────────────────────────
def log_event(pid, condition, event, data):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "participant_id": pid,
        "condition": condition,
        "event": event,
        **data
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── CONDITION ASSIGNMENT ──────────────────────────────────────────────────────
def assign_condition(task_id):
    return "A" if task_id % 2 == 0 else "B"

# ── CONDITION A (KEYWORD SEARCH) ─────────────────────────────────────────────
def keyword_search(query):
    import pandas as pd
    try:
        df = pd.read_csv("small_dataset.csv", encoding="utf-8", encoding_errors="replace")
        q = query.lower()
        mask = df.apply(lambda r: r.astype(str).str.lower().str.contains(q).any(), axis=1)
        return df[mask].head(5).to_dict(orient="records")
    except Exception as e:
        return [{"title": str(e)}]

def search_a(query, state):
    t0 = time.time()

    state["queries"] = state.get("queries", []) + [query]
    state["query_count"] = state.get("query_count", 0) + 1

    results = keyword_search(query)

    log_event(state.get("participant_id"), "A", "search", {
        "query": query,
        "result_count": len(results),
        "elapsed": round(time.time() - t0, 3)
    })

    return f"{len(results)} results found", state

# ── CONDITION B (AI CHAT — OLD GRADIO FORMAT) ────────────────────────────────
SYSTEM_PROMPT = (
    "You are a museum research assistant. Use sources when available. "
    "If unsure, say so clearly."
)

def chat_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    t0 = time.time()

    state["queries"] = state.get("queries", []) + [message]
    state["query_count"] = state.get("query_count", 0) + 1

    # ── Convert old tuple history → Gemini format ──
    contents = []
    for turn in history or []:
        user_msg, bot_msg = turn

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_msg)]
        ))

        contents.append(types.Content(
            role="model",
            parts=[types.Part(text=bot_msg)]
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
                system_instruction=SYSTEM_PROMPT,
                tools=[file_search_tool],
            )
        )

        answer = response.text or "No response generated."

    except Exception as e:
        answer = f"Error: {e}"

    # ── LOG FULL RESPONSE ──
    log_event(state.get("participant_id"), "B", "chat", {
        "query": message,
        "response_text": answer,
        "response_length": len(answer),
        "elapsed": round(time.time() - t0, 3)
    })

    # ── OLD GRADIO FORMAT (IMPORTANT FIX) ──
    new_history = (history or []) + [[message, answer]]

    return "", new_history, state, new_history

# ── SURVEY ────────────────────────────────────────────────────────────────────
def submit_survey(pid, condition, task_id, completed, time_min,
                  final_answer,
                  trust, useful, ease, accuracy_self, recommend,
                  toast_r, toast_c, toast_t,
                  verified, comments, state):

    accuracy_score = grade_answer(task_id, final_answer)

    log_event(pid, condition, "survey", {
        "task_id": task_id,
        "completed": completed,
        "time_min": time_min,
        "final_answer": final_answer,
        "accuracy_score": accuracy_score,
        "trust": trust,
        "usefulness": useful,
        "ease": ease,
        "accuracy_self": accuracy_self,
        "recommend": recommend,
        "toast_reliable": toast_r,
        "toast_confident": toast_c,
        "toast_trustworthy": toast_t,
        "verified": verified,
        "comments": comments,
        "query_count": state.get("query_count", 0),
        "queries": state.get("queries", [])
    })

    return f"Submitted successfully. Accuracy score: {accuracy_score}"

# ── SESSION START ─────────────────────────────────────────────────────────────
def start_session(pid, task_label, state):
    task_id = int(task_label.split(":")[0].replace("Task ", ""))
    task = next(t for t in TASKS if t["id"] == task_id)

    condition = assign_condition(task_id)

    state.update({
        "participant_id": pid,
        "task_id": task_id,
        "condition": condition,
        "query_count": 0,
        "queries": [],
        "start": time.time()
    })

    log_event(pid, condition, "start", {"task_id": task_id})

    return f"""
Task: {task['title']}
Condition: {condition}

Hint: {task['hint']}
""", state

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:

    state = gr.State({})
    chat_history = gr.State([])

    gr.Markdown("# Museum Study (Stable Version)")

    # ── SETUP ──
    with gr.Tab("Setup"):
        pid = gr.Textbox(label="Participant ID")
        task = gr.Dropdown(
            choices=[f"Task {t['id']}: {t['title']}" for t in TASKS]
        )
        btn = gr.Button("Start")
        out = gr.Markdown()

        btn.click(start_session, [pid, task, state], [out, state])

    # ── KEYWORD SEARCH ──
    with gr.Tab("Keyword Search"):
        q = gr.Textbox()
        btn2 = gr.Button("Search")
        out2 = gr.Textbox()

        btn2.click(search_a, [q, state], [out2, state])

    # ── AI CHAT (FIXED FOR YOUR GRADIO VERSION) ──
    with gr.Tab("AI Chat"):

        chatbot = gr.Chatbot(height=450)  # ❗ NO type="messages"

        msg = gr.Textbox()
        send = gr.Button("Send")

        send.click(chat_b, [msg, chat_history, state],
                   [msg, chatbot, state, chat_history])

        msg.submit(chat_b, [msg, chat_history, state],
                  [msg, chatbot, state, chat_history])

    # ── SURVEY ──
    with gr.Tab("Survey"):
        final = gr.Textbox(label="Final answer")

        completed = gr.Radio(["Yes", "Partial", "No"])
        time_min = gr.Number()

        trust = gr.Slider(1, 5)
        useful = gr.Slider(1, 5)
        ease = gr.Slider(1, 5)
        accuracy_self = gr.Slider(1, 5)
        recommend = gr.Slider(1, 5)

        toast_r = gr.Slider(1, 5)
        toast_c = gr.Slider(1, 5)
        toast_t = gr.Slider(1, 5)

        verified = gr.Radio(["Yes", "No"])
        comments = gr.Textbox()

        submit = gr.Button("Submit")

        submit.click(
            submit_survey,
            [pid, gr.State("A"), gr.State(1), completed, time_min,
             final,
             trust, useful, ease, accuracy_self, recommend,
             toast_r, toast_c, toast_t,
             verified, comments, state],
            gr.Markdown()
        )

demo.launch()