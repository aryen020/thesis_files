import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "google-genai"], check=True)

import os
import json
import time
import threading
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
COUNTER_FILE        = "participant_counter.json"   # NEW: persistent P01/P02/… counter
# ─────────────────────────────────────────────────────────────────────────────

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
    },
    {
        "id": 2,
        "title": "Identify the oldest item",
        "description": "What is the oldest item in the dataset? Provide its name, ID, and creation date.",
    },
    {
        "id": 3,
        "title": "Find lacquerware",
        "description": "Find two examples of lacquerware in the collection. Who made them and when?",
    },
    {
        "id": 4,
        "title": "Artworks from 1700",
        "description": "How many items in the collection were created around 1700? List at least three.",
    },
    {
        "id": 5,
        "title": "Anonymous creators",
        "description": "Find three items created by anonymous makers. What types of objects are they?",
    },
]

# ── Thread-safe auto-incrementing participant ID (P01, P02, …) ────────────────
_counter_lock = threading.Lock()

def _read_counter():
    try:
        with open(COUNTER_FILE) as f:
            return json.load(f).get("next", 1)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1

def _write_counter(n):
    with open(COUNTER_FILE, "w") as f:
        json.dump({"next": n}, f)

def generate_participant_id():
    """Returns the next sequential ID like P01, P02, … P99, P100, …"""
    with _counter_lock:
        n = _read_counter()
        _write_counter(n + 1)
    return f"P{n:02d}"

# NEW: expose counter to researcher panel
def get_participant_count():
    return max(0, _read_counter() - 1)

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
                "id":      str(row.get("ID", "")),
                "title":   str(row.get("Title", "Unknown")),
                "type":    str(row.get("Type", "")),
                "creator": str(row.get("Creator", "Anonymous")),
                "date":    str(row.get("Creation_Date", "")),
                "url":     str(row.get("Identifier", "")),
                "image":   str(row.get("Image_URL", "")),
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
        "query":        query,
        "query_length": len(query),
        "query_number": state.get("query_count", 0),
        "result_count": len(results),
        "elapsed_s":    elapsed,
    })

    if not results:
        return "<p style='color:#e07b39;'>No results found. Try a different keyword.</p>", state

    cards = ""
    for r in results:
        img_html = ""
        if r["image"] and r["image"] != "nan":
            img_html = f"<img src='{r['image']}' style='width:72px;height:72px;object-fit:cover;border-radius:8px;flex-shrink:0;' onerror=\"this.style.display='none'\">"
        link = f"<a href='{r['url']}' target='_blank' style='color:#3b82f6;font-size:11px;'>View record ↗</a>" if r["url"] and r["url"] != "nan" else ""
        cards += f"""
<div style='display:flex;gap:12px;align-items:flex-start;background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin-bottom:10px;'>
    {img_html}
    <div style='flex:1;min-width:0;'>
        <div style='font-weight:600;font-size:14px;margin-bottom:3px;'>{r['title']}</div>
        <div style='color:#6b7280;font-size:12px;'>
            Type: {r['type']} &nbsp;·&nbsp; Creator: {r['creator']} &nbsp;·&nbsp; Date: {r['date']}
        </div>
        <div style='margin-top:6px;'>{link}</div>
    </div>
</div>"""

    html = f"<div style='font-size:12px;color:#9ca3af;margin-bottom:10px;'>{len(results)} result(s) for &quot;{query}&quot; · {elapsed}s · Keyword matches only, no AI interpretation.</div>{cards}"
    return html, state

# ── Condition B: RAG Chat ─────────────────────────────────────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum research assistant. Answer questions using the indexed collection documents. "
    "Be clear and informative. If you are unsure or the document does not contain the answer, say so explicitly. "
    "Always mention which source document you used."
)

def _to_gemini_contents(history):
    contents = []
    for turn in (history or []):
        if isinstance(turn, dict):
            role = turn.get("role", "")
            text = str(turn.get("content", ""))
            if role == "user" and text:
                contents.append(types.Content(role="user",  parts=[types.Part(text=text)]))
            elif role == "assistant" and text:
                contents.append(types.Content(role="model", parts=[types.Part(text=text)]))
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, b = turn
            if u:
                contents.append(types.Content(role="user",  parts=[types.Part(text=str(u))]))
            if b:
                contents.append(types.Content(role="model", parts=[types.Part(text=str(b))]))
    return contents

def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"] = state.get("queries", []) + [message]
    t_start = time.time()

    contents = _to_gemini_contents(history)
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
        answer = response.text or "No response generated."

        sources = []
        try:
            for candidate in response.candidates:
                if candidate.grounding_metadata:
                    for chunk in (candidate.grounding_metadata.grounding_chunks or []):
                        rc = getattr(chunk, "retrieved_context", None)
                        if rc:
                            name = getattr(rc, "title", None) or getattr(rc, "uri", None)
                            if name and name not in sources:
                                sources.append(name)
        except Exception:
            pass

        elapsed = round(time.time() - t_start, 3)

        low_confidence_phrases = ["i'm not sure", "i don't know", "cannot find", "not mentioned", "no information"]
        if any(p in answer.lower() for p in low_confidence_phrases):
            answer += "\n\n⚠️ Uncertainty notice: The AI indicated limited confidence. Please verify with the original source."

        if sources:
            src_links = " · ".join(s.replace('.txt','').replace('.pdf','') for s in sources)
            answer += f"\n\nSources: {src_links}"

        log_event(state.get("participant_id", "unknown"), "B", "chat", {
            "query":            message,
            "query_length":     len(message),
            "query_number":     state.get("query_count", 0),
            "response_text":    answer,
            "response_length":  len(answer),
            "sources":          sources,
            "elapsed_s":        elapsed,
        })

    except Exception as e:
        answer = f"Error: {e}"

    new_history = list(history or []) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", new_history, state, new_history

# ── Survey submission ─────────────────────────────────────────────────────────
def submit_pre_survey(pid, age, education, language, museum_familiarity,
                      ai_usage, aias1, aias2, aias3, aias4, search_comfort, state):
    data = {
        "participant_id":     pid,
        "age":                age,
        "education":          education,
        "native_language":    language,
        "museum_familiarity": museum_familiarity,
        "ai_usage_freq":      ai_usage,
        "aias4_item1":        aias1,
        "aias4_item2":        aias2,
        "aias4_item3":        aias3,
        "aias4_item4":        aias4,
        "search_comfort":     search_comfort,
    }
    log_event(pid, state.get("condition", "?"), "pre_survey", data)

def submit_survey(pid, condition, task_id, task_completed, completion_time,
                  q_answer_text, q_confidence,
                  q_toast_reliable, q_toast_confident, q_toast_trustworthy,
                  q_tlx_mental, q_tlx_effort,
                  q_manipulation_check,
                  q_verified, q_comments, state):
    survey_data = {
        "participant_id":         pid,
        "condition":              condition,
        "task_id":                task_id,
        "task_completed":         task_completed,
        "self_reported_time_min": completion_time,
        "participant_answer":     q_answer_text,
        "answer_confidence":      q_confidence,
        "toast_reliable":         q_toast_reliable,
        "toast_confident":        q_toast_confident,
        "toast_trustworthy":      q_toast_trustworthy,
        "tlx_mental_demand":      q_tlx_mental,
        "tlx_effort":             q_tlx_effort,
        "manipulation_check":     q_manipulation_check,
        "verified_sources":       q_verified,
        "comments":               q_comments,
        "query_count":            state.get("query_count", 0),
        "queries":                state.get("queries", []),
    }
    log_event(pid, condition, "survey", survey_data)

def end_session(state):
    elapsed = round(time.time() - state.get("session_start", time.time()), 1)
    log_event(state.get("participant_id", "unknown"), state.get("condition", "?"), "session_end", {
        "task_id":       state.get("task_id"),
        "total_time_s":  elapsed,
        "total_queries": state.get("query_count", 0),
        "all_queries":   state.get("queries", []),
    })

# ── Researcher helpers ────────────────────────────────────────────────────────
def download_log():
    if os.path.exists(LOG_FILE):
        return LOG_FILE
    return None

def load_log():
    try:
        with open(LOG_FILE) as f:
            return f.read()
    except FileNotFoundError:
        return "No log entries yet."

def check_researcher_password(password):
    return password == RESEARCHER_PASSWORD

def log_event_js(participant_id, condition, event_type, data):
    log_event(participant_id, condition, event_type, data if isinstance(data, dict) else {})
    return True

# NEW: called by researcher panel to get a live summary
def get_researcher_stats():
    total = get_participant_count()
    cond_a = cond_b = surveys = 0
    task_counts = {t["id"]: 0 for t in TASKS}
    try:
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("event") == "session_start":
                        if "A" in str(e.get("condition", "")):
                            cond_a += 1
                        else:
                            cond_b += 1
                        tid = e.get("task_id")
                        if tid in task_counts:
                            task_counts[tid] += 1
                    if e.get("event") == "survey":
                        surveys += 1
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return {
        "total_registered": total,
        "condition_a": cond_a,
        "condition_b": cond_b,
        "surveys_submitted": surveys,
        "task_counts": task_counts,
    }

# ── Wizard HTML ───────────────────────────────────────────────────────────────
# Tasks data as JS literal (injected into the HTML)
_TASKS_JS = json.dumps(TASKS)

WIZARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Museum Collection Study</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --c-bg: #f8f7f4;
    --c-surface: #ffffff;
    --c-border: #e5e2da;
    --c-border-strong: #c9c5bb;
    --c-text: #1a1917;
    --c-muted: #6b6860;
    --c-subtle: #9c9890;
    --c-accent: #2563eb;
    --c-accent-light: #eff6ff;
    --c-accent-border: #bfdbfe;
    --c-success-bg: #f0fdf4;
    --c-success-border: #bbf7d0;
    --c-success-text: #15803d;
    --c-warn-bg: #fffbeb;
    --c-warn-border: #fde68a;
    --c-warn-text: #92400e;
    --radius: 10px;
    --radius-sm: 6px;
  }
  body { font-family: 'Georgia', serif; background: var(--c-bg); color: var(--c-text); min-height: 100vh; }
  .layout { max-width: 680px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  .study-header { text-align: center; margin-bottom: 2.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--c-border); }
  .study-header h1 { font-size: 1.1rem; font-weight: normal; letter-spacing: 0.05em; text-transform: uppercase; color: var(--c-muted); margin-bottom: 0.25rem; }
  .study-header p { font-size: 0.8rem; color: var(--c-subtle); font-family: 'Courier New', monospace; }

  .progress-wrap { margin-bottom: 2rem; }
  .progress-steps { display: flex; align-items: center; }
  .progress-step { flex: 1; height: 3px; background: var(--c-border); transition: background 0.4s; }
  .progress-step.done { background: var(--c-accent); }
  .progress-step.active { background: var(--c-text); }
  .progress-label { display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.7rem; font-family: 'Courier New', monospace; color: var(--c-subtle); }

  .step { display: none; animation: fadeUp 0.35s ease; }
  .step.active { display: block; }
  @keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

  .step-eyebrow { font-family: 'Courier New', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--c-muted); margin-bottom: 0.5rem; }
  .step-title { font-size: 1.6rem; font-weight: normal; margin-bottom: 0.6rem; line-height: 1.25; }
  .step-sub { font-size: 0.9rem; color: var(--c-muted); line-height: 1.65; margin-bottom: 1.75rem; }

  .card { background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .card-info { background: var(--c-accent-light); border-color: var(--c-accent-border); }
  .card-success { background: var(--c-success-bg); border-color: var(--c-success-border); }
  .card-warn { background: var(--c-warn-bg); border-color: var(--c-warn-border); font-size: 0.85rem; color: var(--c-warn-text); }

  .field { margin-bottom: 1.25rem; }
  .field label { display: block; font-size: 0.8rem; font-family: 'Courier New', monospace; text-transform: uppercase; letter-spacing: 0.04em; color: var(--c-muted); margin-bottom: 0.4rem; }
  .field input[type=text], .field input[type=number], .field select, .field textarea {
    width: 100%; padding: 0.6rem 0.85rem; font-size: 0.95rem; font-family: 'Georgia', serif;
    background: var(--c-bg); border: 1px solid var(--c-border); border-radius: var(--radius-sm);
    color: var(--c-text); outline: none; transition: border-color 0.2s, box-shadow 0.2s; -webkit-appearance: none;
  }
  .field input:focus, .field select:focus, .field textarea:focus { border-color: var(--c-accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
  .field textarea { resize: vertical; min-height: 90px; line-height: 1.6; }
  .field select { cursor: pointer; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b6860' d='M6 8L1 3h10z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 10px center; padding-right: 2rem; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 480px) { .two-col { grid-template-columns: 1fr; } }

  .slider-field { margin-bottom: 1.4rem; }
  .slider-label { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
  .slider-label span:first-child { font-size: 0.8rem; font-family: 'Courier New', monospace; text-transform: uppercase; letter-spacing: 0.04em; color: var(--c-muted); }
  .slider-val { font-size: 1rem; font-weight: bold; font-family: 'Courier New', monospace; color: var(--c-text); }
  input[type=range] { -webkit-appearance: none; width: 100%; height: 4px; background: var(--c-border); border-radius: 2px; outline: none; cursor: pointer; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: var(--c-text); border: 2px solid var(--c-surface); box-shadow: 0 1px 3px rgba(0,0,0,0.2); cursor: pointer; transition: transform 0.1s; }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.15); }
  .slider-ticks { display: flex; justify-content: space-between; font-size: 0.7rem; font-family: 'Courier New', monospace; color: var(--c-subtle); margin-top: 4px; }

  .radio-group { display: flex; flex-direction: column; gap: 8px; }
  .radio-opt { display: flex; align-items: flex-start; gap: 10px; padding: 0.75rem 1rem; border: 1px solid var(--c-border); border-radius: var(--radius-sm); cursor: pointer; background: var(--c-surface); transition: border-color 0.2s, background 0.2s; font-size: 0.9rem; line-height: 1.4; }
  .radio-opt:hover { border-color: var(--c-border-strong); }
  .radio-opt.selected { border-color: var(--c-accent); background: var(--c-accent-light); }
  .radio-opt input[type=radio] { margin-top: 2px; flex-shrink: 0; accent-color: var(--c-accent); }

  .btn { display: inline-flex; align-items: center; gap: 6px; padding: 0.6rem 1.25rem; font-size: 0.9rem; font-family: 'Georgia', serif; border-radius: var(--radius-sm); border: 1px solid var(--c-border-strong); background: transparent; color: var(--c-text); cursor: pointer; transition: background 0.15s, transform 0.1s; text-decoration: none; }
  .btn:hover { background: var(--c-border); }
  .btn:active { transform: scale(0.98); }
  .btn-primary { background: var(--c-text); color: #fff; border-color: transparent; }
  .btn-primary:hover { opacity: 0.85; background: var(--c-text); }
  .btn-nav { display: flex; justify-content: space-between; align-items: center; margin-top: 2rem; padding-top: 1.25rem; border-top: 1px solid var(--c-border); }

  .section-label { font-family: 'Courier New', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--c-muted); margin: 1.75rem 0 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--c-border); }

  .pid-badge { display: inline-flex; align-items: center; gap: 8px; background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 999px; padding: 0.3rem 1rem; font-family: 'Courier New', monospace; font-size: 0.8rem; color: var(--c-muted); margin-bottom: 1.25rem; }
  .pid-badge strong { color: var(--c-text); }

  .search-row { display: flex; gap: 8px; margin-bottom: 1rem; }
  .search-row input { flex: 1; padding: 0.6rem 0.85rem; font-size: 0.95rem; font-family: 'Georgia', serif; background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius-sm); color: var(--c-text); outline: none; transition: border-color 0.2s; }
  .search-row input:focus { border-color: var(--c-accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }

  .result-item { display: flex; gap: 12px; align-items: flex-start; background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius-sm); padding: 14px; margin-bottom: 10px; }
  .result-img { width: 68px; height: 68px; object-fit: cover; border-radius: 6px; flex-shrink: 0; background: var(--c-border); }
  .result-body { flex: 1; min-width: 0; }
  .result-title { font-weight: bold; font-size: 0.9rem; margin-bottom: 3px; }
  .result-meta { font-size: 0.78rem; color: var(--c-muted); font-family: 'Courier New', monospace; }
  .result-link { font-size: 0.78rem; color: var(--c-accent); text-decoration: none; margin-top: 5px; display: inline-block; }
  .result-link:hover { text-decoration: underline; }

  .chat-window { background: var(--c-bg); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 1rem; min-height: 240px; max-height: 360px; overflow-y: auto; margin-bottom: 10px; }
  .msg { margin-bottom: 14px; }
  .msg-role { font-family: 'Courier New', monospace; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--c-subtle); margin-bottom: 4px; }
  .msg-user { text-align: right; }
  .msg-bubble { display: inline-block; padding: 0.6rem 0.9rem; border-radius: 12px; font-size: 0.9rem; line-height: 1.55; max-width: 88%; text-align: left; }
  .msg-user .msg-bubble { background: var(--c-text); color: #fff; border-bottom-right-radius: 3px; }
  .msg-ai .msg-bubble { background: var(--c-surface); border: 1px solid var(--c-border); border-bottom-left-radius: 3px; white-space: pre-wrap; }
  .chat-row { display: flex; gap: 8px; }
  .chat-row input { flex: 1; padding: 0.6rem 0.85rem; font-size: 0.9rem; font-family: 'Georgia', serif; background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius-sm); color: var(--c-text); outline: none; transition: border-color 0.2s; }
  .chat-row input:focus { border-color: var(--c-accent); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }

  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--c-border); border-top-color: var(--c-muted); border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: -3px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Task reference list (participant view — all tasks shown during search step) */
  .all-tasks { margin-top: 1.5rem; }
  .task-ref { border: 1px solid var(--c-border); border-radius: var(--radius-sm); padding: 0.75rem 1rem; margin-bottom: 8px; background: var(--c-surface); }
  .task-ref-header { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
  .task-num { font-family: 'Courier New', monospace; font-size: 0.7rem; background: var(--c-bg); border: 1px solid var(--c-border); border-radius: 4px; padding: 1px 6px; color: var(--c-muted); }
  .task-ref-title { font-size: 0.85rem; font-weight: bold; }
  .task-ref-desc { font-size: 0.82rem; color: var(--c-muted); line-height: 1.5; }
  .task-ref.current { border-color: var(--c-accent); background: var(--c-accent-light); }
  .task-ref.current .task-num { background: var(--c-accent); color: #fff; border-color: transparent; }

  .done-screen { text-align: center; padding: 3rem 1rem; }
  .done-icon { width: 60px; height: 60px; border-radius: 50%; background: var(--c-success-bg); border: 1px solid var(--c-success-border); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin: 0 auto 1.5rem; }
  .done-meta { display: inline-grid; grid-template-columns: auto auto; gap: 4px 20px; text-align: left; margin-top: 1.5rem; font-size: 0.85rem; }
  .done-meta dt { color: var(--c-muted); font-family: 'Courier New', monospace; text-transform: uppercase; font-size: 0.7rem; align-self: end; }
  .done-meta dd { font-weight: bold; font-family: 'Courier New', monospace; }

  .log-box { background: #1a1917; color: #d4d0c8; font-family: 'Courier New', monospace; font-size: 0.75rem; padding: 1rem; border-radius: var(--radius-sm); max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; margin-top: 1rem; }

  .feature-list { list-style: none; }
  .feature-list li { display: flex; gap: 10px; align-items: flex-start; font-size: 0.9rem; color: var(--c-muted); padding: 6px 0; border-bottom: 1px solid var(--c-border); }
  .feature-list li:last-child { border-bottom: none; }
  .feature-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

  /* Researcher panel */
  .r-stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 1rem; }
  .r-stat { background: #f1f0ed; border-radius: 6px; padding: 10px 12px; text-align: center; }
  .r-stat-n { font-family: 'Courier New', monospace; font-size: 1.4rem; font-weight: bold; }
  .r-stat-l { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b6860; margin-top: 2px; }
  .r-task-row { display: flex; gap: 8px; align-items: baseline; border-bottom: 1px solid #e5e2da; padding: 6px 0; font-size: 0.82rem; }
  .r-task-id { font-family: 'Courier New', monospace; font-weight: bold; min-width: 30px; }
  .r-task-name { flex: 1; color: #444; }
  .r-task-count { font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; }
  .r-gt-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin-top: 4px; margin-bottom: 2px; }
  .r-gt-input { width: 100%; font-size: 0.8rem; font-family: 'Georgia', serif; padding: 4px 8px; border: 1px solid #e5e2da; border-radius: 4px; background: #fafaf8; color: #1a1917; margin-top: 2px; }
  .r-search-row { display: flex; gap: 6px; margin-bottom: 8px; }
  .r-search-row input { flex: 1; font-size: 0.8rem; padding: 5px 8px; border: 1px solid #e5e2da; border-radius: 4px; background: #fafaf8; color: #1a1917; outline: none; }
  .r-tab { cursor: pointer; padding: 4px 10px; font-size: 0.75rem; font-family: 'Courier New', monospace; border: 1px solid #e5e2da; border-radius: 4px; background: transparent; }
  .r-tab.active { background: #1a1917; color: #fff; border-color: transparent; }
</style>
</head>
<body>
<div class="layout">
  <div class="study-header">
    <h1>Museum Collection Study</h1>
    <p id="pid-header">Loading…</p>
  </div>

  <div class="progress-wrap">
    <div class="progress-steps" id="progress-steps"></div>
    <div class="progress-label" id="progress-label"></div>
  </div>

  <!-- ══ STEP 0: Welcome ══════════════════════════════════════════════ -->
  <div class="step active" id="step-0">
    <div class="step-eyebrow">Welcome</div>
    <div class="step-title">Welcome to the study</div>
    <div class="step-sub">This study looks at how people search museum collections using different tools. It takes about 15–20 minutes and is fully anonymous.</div>
    <div class="card">
      <ul class="feature-list">
        <li><span class="feature-icon">⏱</span> About 15–20 minutes from start to finish</li>
        <li><span class="feature-icon">💻</span> Works on any device — fully remote</li>
        <li><span class="feature-icon">🔒</span> Completely anonymous — no names collected</li>
        <li><span class="feature-icon">➡️</span> Follow the steps — press Next at each stage</li>
      </ul>
    </div>
    <div class="card-warn card" style="margin-top:0.75rem;">
      Please do not refresh the page during the study. Your progress will be lost.
    </div>
    <div class="btn-nav" style="border-top:none;padding-top:0;margin-top:1.5rem;">
      <div></div>
      <button class="btn btn-primary" onclick="goTo(1)">Start study &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 1: Demographics ════════════════════════════════════════ -->
  <div class="step" id="step-1">
    <div class="step-eyebrow">Step 1 of 6 — Background</div>
    <div class="step-title">A bit about you</div>
    <div class="step-sub">This helps us account for individual differences in the analysis. All responses are anonymous.</div>

    <div class="two-col">
      <div class="field"><label>Age</label><input type="number" id="pre_age" min="18" max="99" placeholder="e.g. 28"></div>
      <div class="field"><label>Native language</label><input type="text" id="pre_lang" placeholder="e.g. Dutch"></div>
    </div>
    <div class="field">
      <label>Highest education level</label>
      <select id="pre_edu">
        <option value="">Select…</option>
        <option>Secondary / high school</option>
        <option>Bachelor's degree</option>
        <option>Master's degree</option>
        <option>PhD / doctorate</option>
        <option>Other</option>
      </select>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>Museum / art history familiarity</span><span class="slider-val" id="museum_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="pre_museum" oninput="document.getElementById('museum_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — None</span><span>5 — Expert</span></div>
    </div>
    <div class="field">
      <label>How often do you use ChatGPT-style AI tools?</label>
      <select id="pre_ai">
        <option value="">Select…</option>
        <option>Never</option>
        <option>Rarely (a few times a year)</option>
        <option>Sometimes (monthly)</option>
        <option>Often (weekly)</option>
        <option>Daily</option>
      </select>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>Comfort searching databases / catalogues</span><span class="slider-val" id="search_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="pre_search" oninput="document.getElementById('search_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Not at all</span><span>5 — Very comfortable</span></div>
    </div>
    <div class="btn-nav">
      <button class="btn" onclick="goTo(0)">&larr; Back</button>
      <button class="btn btn-primary" onclick="goTo(2)">Next &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 2: AIAS-4 ══════════════════════════════════════════════ -->
  <div class="step" id="step-2">
    <div class="step-eyebrow">Step 2 of 6 — AI Attitudes</div>
    <div class="step-title">Your views on AI</div>
    <div class="step-sub">Rate each statement from 1 (strongly disagree) to 5 (strongly agree). (AIAS-4, Grassini 2023)</div>

    <div class="slider-field">
      <div class="slider-label"><span>AI systems can perform tasks as well as humans</span><span class="slider-val" id="aias1_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="aias1" oninput="document.getElementById('aias1_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>5 — Strongly agree</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>I feel comfortable relying on AI for information</span><span class="slider-val" id="aias2_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="aias2" oninput="document.getElementById('aias2_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>5 — Strongly agree</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>AI tools are a useful addition to everyday work</span><span class="slider-val" id="aias3_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="aias3" oninput="document.getElementById('aias3_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>5 — Strongly agree</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>I trust AI-generated results to be mostly accurate</span><span class="slider-val" id="aias4_val">3</span></div>
      <input type="range" min="1" max="5" step="1" value="3" id="aias4" oninput="document.getElementById('aias4_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>5 — Strongly agree</span></div>
    </div>
    <div class="btn-nav">
      <button class="btn" onclick="goTo(1)">&larr; Back</button>
      <button class="btn btn-primary" onclick="savePreSurveyAndContinue()">Next &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 3: Task assignment ══════════════════════════════════════ -->
  <div class="step" id="step-3">
    <div class="step-eyebrow">Step 3 of 6 — Your Task</div>
    <div class="step-title">Your assigned task</div>
    <div class="step-sub">You have been randomly assigned to a search tool and a task. Read the task carefully before you begin.</div>

    <div class="pid-badge">&#9679; Participant ID: <strong id="pid-display">—</strong></div>

    <div class="card card-info" style="margin-bottom:1rem;">
      <div style="font-size:0.75rem;font-family:'Courier New',monospace;text-transform:uppercase;letter-spacing:0.05em;color:#1d4ed8;margin-bottom:4px;">Your assigned tool</div>
      <div id="condition-text" style="font-size:0.95rem;color:#1e40af;"></div>
    </div>

    <div class="section-label" style="margin-top:0;">Your task</div>
    <div class="card">
      <div id="task-title-display" style="font-size:1rem;font-weight:bold;margin-bottom:6px;"></div>
      <div id="task-desc-display" style="font-size:0.9rem;color:#4b5563;line-height:1.6;"></div>
    </div>

    <div class="btn-nav">
      <button class="btn" onclick="goTo(2)">&larr; Back</button>
      <button class="btn btn-primary" onclick="startTask()">Begin task &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 4A: Keyword Search ══════════════════════════════════════ -->
  <div class="step" id="step-4a">
    <div class="step-eyebrow">Step 4 of 6 — Search</div>
    <div class="step-title">Keyword search</div>
    <div class="step-sub">Type keywords to search the museum collection. Results are direct matches only — no AI interpretation.</div>

    <div class="card card-info" style="padding:0.9rem 1.1rem;margin-bottom:1rem;">
      <div style="font-size:0.72rem;font-family:'Courier New',monospace;text-transform:uppercase;letter-spacing:0.05em;color:#1d4ed8;margin-bottom:3px;">Your assigned task</div>
      <div id="task-reminder-a" style="font-size:0.88rem;color:#1e40af;line-height:1.55;font-weight:bold;"></div>
    </div>

    <div class="search-row">
      <input type="text" id="search_input" placeholder="e.g. ritual, lacquer, 1700, anonymous…" onkeydown="if(event.key==='Enter')doSearch()">
      <button class="btn btn-primary" onclick="doSearch()">Search</button>
    </div>
    <div id="search_results"></div>

    <!-- ALL TASKS listed for reference during search -->
    <div class="all-tasks">
      <div class="section-label">All tasks — for reference</div>
      <div id="all-tasks-a"></div>
    </div>

    <div class="btn-nav">
      <button class="btn" onclick="goTo(3)">&larr; Back</button>
      <button class="btn btn-primary" onclick="goTo(5)">I found my answer &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 4B: AI Chat ══════════════════════════════════════════════ -->
  <div class="step" id="step-4b">
    <div class="step-eyebrow">Step 4 of 6 — AI Assistant</div>
    <div class="step-title">AI research assistant</div>
    <div class="step-sub">Ask questions in natural language. AI answers may contain errors — always verify important details with source links.</div>

    <div class="card card-info" style="padding:0.9rem 1.1rem;margin-bottom:1rem;">
      <div style="font-size:0.72rem;font-family:'Courier New',monospace;text-transform:uppercase;letter-spacing:0.05em;color:#1d4ed8;margin-bottom:3px;">Your assigned task</div>
      <div id="task-reminder-b" style="font-size:0.88rem;color:#1e40af;line-height:1.55;font-weight:bold;"></div>
    </div>

    <div class="chat-window" id="chat_messages">
      <div class="msg msg-ai">
        <div class="msg-role">Assistant</div>
        <div class="msg-bubble">Hi! I can help you search the museum collection. What would you like to find?</div>
      </div>
    </div>
    <div class="chat-row">
      <input type="text" id="chat_input" placeholder="Ask about the collection…" onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn btn-primary" onclick="sendChat()">Send</button>
    </div>

    <!-- ALL TASKS listed for reference during chat -->
    <div class="all-tasks">
      <div class="section-label">All tasks — for reference</div>
      <div id="all-tasks-b"></div>
    </div>

    <div class="btn-nav">
      <button class="btn" onclick="goTo(3)">&larr; Back</button>
      <button class="btn btn-primary" onclick="goTo(5)">I found my answer &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 5: Post-task survey ════════════════════════════════════ -->
  <div class="step" id="step-5">
    <div class="step-eyebrow">Step 5 of 6 — Survey</div>
    <div class="step-title">Quick survey</div>
    <div class="step-sub">A few questions about your experience. This takes about 3 minutes.</div>

    <div class="section-label" style="margin-top:0;">Your answer (H2 — Accuracy)</div>
    <div class="field">
      <label>What is your final answer to the task? Write it out fully.</label>
      <textarea id="s_answer" placeholder="e.g. The ritual object is called X, made in year Y, from culture Z."></textarea>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>How confident are you that your answer is correct?</span><span class="slider-val" id="conf_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="s_confidence" oninput="document.getElementById('conf_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Not at all sure</span><span>7 — Completely sure</span></div>
    </div>

    <div class="section-label">TOAST trust scale (H3)</div>
    <div class="slider-field">
      <div class="slider-label"><span>The system performed reliably</span><span class="slider-val" id="toast1_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="toast_reliable" oninput="document.getElementById('toast1_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>7 — Strongly agree</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>I felt confident using this system</span><span class="slider-val" id="toast2_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="toast_confident" oninput="document.getElementById('toast2_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>7 — Strongly agree</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>I found the system trustworthy</span><span class="slider-val" id="toast3_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="toast_trustworthy" oninput="document.getElementById('toast3_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Strongly disagree</span><span>7 — Strongly agree</span></div>
    </div>

    <div class="section-label">Cognitive load — NASA-TLX (H4)</div>
    <div class="slider-field">
      <div class="slider-label"><span>How mentally demanding was the task?</span><span class="slider-val" id="tlx1_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="tlx_mental" oninput="document.getElementById('tlx1_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Not at all</span><span>7 — Extremely</span></div>
    </div>
    <div class="slider-field">
      <div class="slider-label"><span>How hard did you have to work to accomplish your performance?</span><span class="slider-val" id="tlx2_val">4</span></div>
      <input type="range" min="1" max="7" step="1" value="4" id="tlx_effort" oninput="document.getElementById('tlx2_val').textContent=this.value">
      <div class="slider-ticks"><span>1 — Very little</span><span>7 — Very hard</span></div>
    </div>

    <div class="section-label">Task completion</div>
    <div class="field">
      <label>Did you complete the task?</label>
      <div class="radio-group" id="grp-completed">
        <div class="radio-opt" onclick="pick('grp-completed',this,'Yes')"><input type="radio" name="completed"> Yes, I found the answer</div>
        <div class="radio-opt" onclick="pick('grp-completed',this,'Partially')"><input type="radio" name="completed"> Partially — I found something but I'm not certain</div>
        <div class="radio-opt" onclick="pick('grp-completed',this,'No')"><input type="radio" name="completed"> No, I couldn't find the answer</div>
      </div>
    </div>
    <div class="field">
      <label>Approximate time taken (minutes)</label>
      <input type="number" id="s_time" min="1" max="120" value="5">
    </div>

    <div class="section-label">Manipulation check</div>
    <div class="field">
      <label>What kind of tool did you just use?</label>
      <div class="radio-group" id="grp-manip">
        <div class="radio-opt" onclick="pick('grp-manip',this,'Keyword search')"><input type="radio" name="manip"> Keyword search (basic matching)</div>
        <div class="radio-opt" onclick="pick('grp-manip',this,'AI assistant')"><input type="radio" name="manip"> AI assistant (natural language)</div>
        <div class="radio-opt" onclick="pick('grp-manip',this,'Both')"><input type="radio" name="manip"> Both</div>
        <div class="radio-opt" onclick="pick('grp-manip',this,'Not sure')"><input type="radio" name="manip"> Not sure</div>
      </div>
    </div>

    <div class="section-label">Critical evaluation</div>
    <div class="field">
      <label>Did you verify any answers with the source link?</label>
      <div class="radio-group" id="grp-verified">
        <div class="radio-opt" onclick="pick('grp-verified',this,'Yes, always')"><input type="radio" name="verified"> Yes, always</div>
        <div class="radio-opt" onclick="pick('grp-verified',this,'Sometimes')"><input type="radio" name="verified"> Sometimes</div>
        <div class="radio-opt" onclick="pick('grp-verified',this,'No')"><input type="radio" name="verified"> No</div>
      </div>
    </div>
    <div class="field">
      <label>Comments or feedback (optional)</label>
      <textarea id="s_comments" placeholder="What worked well? What was frustrating?"></textarea>
    </div>

    <div class="btn-nav">
      <button class="btn" id="back-to-task">&larr; Back to task</button>
      <button class="btn btn-primary" onclick="submitAll()">Submit survey &rarr;</button>
    </div>
  </div>

  <!-- ══ STEP 6: Done ═════════════════════════════════════════════════ -->
  <div class="step" id="step-6">
    <div class="done-screen">
      <div class="done-icon">✓</div>
      <div class="step-title">Thank you!</div>
      <div class="step-sub" style="margin-bottom:0;">Your responses have been saved. You may now close this window.</div>
      <dl class="done-meta">
        <dt>Participant ID</dt><dd id="done-pid">—</dd>
        <dt>Condition</dt><dd id="done-cond">—</dd>
        <dt>Task</dt><dd id="done-task">—</dd>
        <dt>Queries made</dt><dd id="done-queries">—</dd>
        <dt>Total time</dt><dd id="done-time">—</dd>
      </dl>
    </div>
  </div>

</div><!-- /layout -->

<!-- ══ Researcher panel (floating, password-protected) ═══════════════════ -->
<div style="position:fixed;bottom:16px;right:16px;z-index:1000;">
  <button class="btn" onclick="toggleResearcher()" style="font-size:0.75rem;padding:0.4rem 0.8rem;background:rgba(255,255,255,0.92);">
    🔬 Researcher
  </button>
</div>

<div id="researcher-modal" style="display:none;position:fixed;bottom:60px;right:16px;width:min(540px,95vw);
     background:#fff;border:1px solid #e5e2da;border-radius:12px;
     box-shadow:0 8px 32px rgba(0,0,0,0.13);z-index:1001;overflow:hidden;">

  <div style="padding:1rem 1.25rem;border-bottom:1px solid #e5e2da;display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'Courier New',monospace;font-size:0.82rem;font-weight:bold;">🔬 Researcher Panel</span>
    <button onclick="toggleResearcher()" style="background:none;border:none;cursor:pointer;font-size:1rem;color:#888;">✕</button>
  </div>

  <!-- Locked view -->
  <div id="researcher-locked" style="padding:1.25rem;">
    <div style="font-size:0.85rem;color:#6b6860;margin-bottom:10px;">Enter your researcher password to access study data.</div>
    <input type="password" id="r_password" placeholder="Password" onkeydown="if(event.key==='Enter')unlockResearcher()"
           style="width:100%;padding:0.5rem 0.75rem;font-size:0.9rem;border:1px solid #e5e2da;border-radius:6px;margin-bottom:8px;outline:none;font-family:'Georgia',serif;">
    <button class="btn btn-primary" onclick="unlockResearcher()" style="width:100%;">Unlock</button>
    <div id="r_error" style="color:#dc2626;font-size:0.8rem;margin-top:6px;"></div>
  </div>

  <!-- Unlocked view -->
  <div id="researcher-open" style="display:none;">
    <!-- Tabs -->
    <div style="display:flex;gap:6px;padding:10px 1.25rem;border-bottom:1px solid #e5e2da;background:#fafaf8;">
      <button class="r-tab active" id="rtab-stats" onclick="showRTab('stats')">Stats</button>
      <button class="r-tab" id="rtab-tasks" onclick="showRTab('tasks')">Tasks &amp; Answers</button>
      <button class="r-tab" id="rtab-log" onclick="showRTab('log')">Log</button>
    </div>

    <!-- Stats tab -->
    <div id="rpanel-stats" style="padding:1.25rem;">
      <div class="r-stat-grid" id="r-stat-grid">
        <div class="r-stat"><div class="r-stat-n" id="rs-total">—</div><div class="r-stat-l">Total</div></div>
        <div class="r-stat"><div class="r-stat-n" id="rs-a">—</div><div class="r-stat-l">Cond. A</div></div>
        <div class="r-stat"><div class="r-stat-n" id="rs-b">—</div><div class="r-stat-l">Cond. B</div></div>
        <div class="r-stat"><div class="r-stat-n" id="rs-surveys">—</div><div class="r-stat-l">Surveys</div></div>
      </div>
      <div style="font-size:0.75rem;color:#888;font-family:'Courier New',monospace;margin-bottom:8px;">Task distribution</div>
      <div id="r-task-dist"></div>
      <button class="btn" onclick="loadStats()" style="margin-top:1rem;font-size:0.8rem;width:100%;">↻ Refresh stats</button>
    </div>

    <!-- Tasks & Answers tab (researcher-only ground-truth) -->
    <div id="rpanel-tasks" style="display:none;padding:1.25rem;max-height:420px;overflow-y:auto;">
      <div style="font-size:0.75rem;color:#888;margin-bottom:12px;line-height:1.5;">
        All five tasks with verified correct answers for H2 accuracy grading. Fill in the ground-truth from your dataset.
        Answers are saved in your browser only — add them to your analysis notes.
      </div>
      <div id="r-tasks-list"></div>
    </div>

    <!-- Log tab -->
    <div id="rpanel-log" style="display:none;padding:1.25rem;">
      <div class="r-search-row">
        <input type="text" id="r-log-search" placeholder="Filter log by participant, event, condition…" oninput="filterLog()">
        <button class="btn" onclick="refreshLog()" style="font-size:0.8rem;">↻</button>
        <button class="btn btn-primary" onclick="downloadLog()" style="font-size:0.8rem;">⬇ Download</button>
      </div>
      <div class="log-box" id="r_log_display">Loading…</div>
    </div>
  </div>
</div>

<script>
const TASKS_DATA = """ + _TASKS_JS + """;
const CONDITIONS = ["A — Keyword Search", "B — AI Chat"];
const STEP_NAMES = ["Welcome","Background","AI Views","Task","Search","Survey","Done"];

let S = {
  pid: "", condition: "", task: null,
  sessionStart: null, queryCount: 0, queries: [],
  chatHistory: [], radios: {},
};

// ── Init: get auto-incremented ID from server ──────────────────────────
async function init() {
  try {
    const resp = await fetch('/run/generate_participant_id', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [] })
    });
    const json = await resp.json();
    S.pid = json.data[0];
  } catch(e) {
    // Fallback: timestamp-based if server unreachable
    S.pid = 'P-' + Date.now().toString(36).toUpperCase();
  }

  S.sessionStart = Date.now();
  S.condition = CONDITIONS[Math.floor(Math.random() * 2)];
  S.task = TASKS_DATA[Math.floor(Math.random() * TASKS_DATA.length)];

  document.getElementById('pid-header').textContent = 'ID: ' + S.pid;
  document.getElementById('pid-display').textContent = S.pid;

  const isA = S.condition.includes('A');
  document.getElementById('condition-text').textContent = isA
    ? '🔍 Keyword search — type keywords to find items by direct matching.'
    : '🤖 AI assistant — ask questions in natural language.';

  document.getElementById('task-title-display').textContent = 'Task ' + S.task.id + ': ' + S.task.title;
  document.getElementById('task-desc-display').textContent = S.task.description;
  document.getElementById('task-reminder-a').textContent = S.task.description;
  document.getElementById('task-reminder-b').textContent = S.task.description;
  document.getElementById('back-to-task').onclick = () => goTo(isA ? '4a' : '4b');

  // Render all-tasks reference list for both search steps
  renderAllTasks('all-tasks-a');
  renderAllTasks('all-tasks-b');

  updateProgress(0);
  logToServer('session_init', { task_id: S.task.id, condition: S.condition });
}

function renderAllTasks(containerId) {
  const el = document.getElementById(containerId);
  el.innerHTML = TASKS_DATA.map(t => {
    const isCurrent = t.id === S.task.id;
    return `<div class="task-ref ${isCurrent ? 'current' : ''}">
      <div class="task-ref-header">
        <span class="task-num">T${t.id}</span>
        <span class="task-ref-title">${t.title}${isCurrent ? ' ← your task' : ''}</span>
      </div>
      <div class="task-ref-desc">${t.description}</div>
    </div>`;
  }).join('');
}

// ── Navigation ────────────────────────────────────────────────────────
function goTo(step) {
  document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
  const map = {0:'step-0',1:'step-1',2:'step-2',3:'step-3','4a':'step-4a','4b':'step-4b',5:'step-5',6:'step-6'};
  const el = document.getElementById(map[step]);
  if (el) el.classList.add('active');
  const numericStep = typeof step === 'number' ? Math.min(step, 6) : 4;
  updateProgress(numericStep);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateProgress(active) {
  const bar = document.getElementById('progress-steps');
  const lbl = document.getElementById('progress-label');
  const n = STEP_NAMES.length;
  bar.innerHTML = STEP_NAMES.map((_,i) => `<div class="progress-step ${i<active?'done':i===active?'active':''}"></div>`).join('');
  lbl.innerHTML = `<span>${STEP_NAMES[0]}</span><span style="font-weight:bold">${STEP_NAMES[Math.min(active,n-1)]}</span><span>${STEP_NAMES[n-1]}</span>`;
}

// ── Radio helper ──────────────────────────────────────────────────────
function pick(groupId, el, value) {
  document.querySelectorAll('#' + groupId + ' .radio-opt').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
  S.radios[groupId] = value;
}

// ── Save pre-survey ───────────────────────────────────────────────────
async function savePreSurveyAndContinue() {
  const data = {
    participant_id: S.pid, age: document.getElementById('pre_age').value,
    education: document.getElementById('pre_edu').value,
    native_language: document.getElementById('pre_lang').value,
    museum_familiarity: document.getElementById('pre_museum').value,
    ai_usage_freq: document.getElementById('pre_ai').value,
    search_comfort: document.getElementById('pre_search').value,
    aias4_item1: document.getElementById('aias1').value,
    aias4_item2: document.getElementById('aias2').value,
    aias4_item3: document.getElementById('aias3').value,
    aias4_item4: document.getElementById('aias4').value,
  };
  logToServer('pre_survey', data);
  goTo(3);
}

// ── Start task ────────────────────────────────────────────────────────
function startTask() {
  logToServer('session_start', { task_id: S.task.id, condition: S.condition });
  goTo(S.condition.includes('A') ? '4a' : '4b');
}

// ── Keyword search ────────────────────────────────────────────────────
async function doSearch() {
  const q = document.getElementById('search_input').value.trim();
  if (!q) return;
  S.queryCount++; S.queries.push(q);
  const el = document.getElementById('search_results');
  el.innerHTML = '<div style="padding:12px 0;color:#9ca3af;font-size:0.85rem;"><span class="spinner"></span> Searching…</div>';
  try {
    const resp = await fetch('/run/search_condition_a', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [q, { participant_id: S.pid, condition: S.condition, query_count: S.queryCount, queries: S.queries }] })
    });
    const json = await resp.json();
    el.innerHTML = json.data[0];
    const st = json.data[1];
    if (st) { S.queryCount = st.query_count || S.queryCount; S.queries = st.queries || S.queries; }
  } catch(e) {
    el.innerHTML = '<div style="color:#dc2626;font-size:0.85rem;">Error: ' + e.message + '</div>';
  }
}

// ── AI chat ───────────────────────────────────────────────────────────
async function sendChat() {
  const input = document.getElementById('chat_input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  S.queryCount++; S.queries.push(msg);
  const chatEl = document.getElementById('chat_messages');
  chatEl.innerHTML += '<div class="msg msg-user"><div class="msg-role">You</div><div class="msg-bubble">' + escHtml(msg) + '</div></div>';
  chatEl.innerHTML += '<div class="msg msg-ai" id="typing-ind"><div class="msg-role">Assistant</div><div class="msg-bubble"><span class="spinner"></span></div></div>';
  chatEl.scrollTop = chatEl.scrollHeight;
  try {
    const resp = await fetch('/run/chat_condition_b', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [msg, S.chatHistory, { participant_id: S.pid, condition: S.condition, query_count: S.queryCount, queries: S.queries }] })
    });
    const json = await resp.json();
    const newHistory = json.data[1];
    const newState = json.data[2];
    S.chatHistory = newHistory;
    if (newState) { S.queryCount = newState.query_count || S.queryCount; S.queries = newState.queries || S.queries; }
    const lastMsg = [...newHistory].reverse().find(m => m.role === 'assistant');
    const answer = lastMsg ? lastMsg.content : '(no response)';
    document.getElementById('typing-ind').outerHTML = '<div class="msg msg-ai"><div class="msg-role">Assistant</div><div class="msg-bubble">' + escHtml(answer) + '</div></div>';
  } catch(e) {
    document.getElementById('typing-ind').outerHTML = '<div class="msg msg-ai"><div class="msg-role">Assistant</div><div class="msg-bubble" style="color:#dc2626;">Error: ' + e.message + '</div></div>';
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ── Submit all ────────────────────────────────────────────────────────
async function submitAll() {
  const totalMin = Math.round((Date.now() - S.sessionStart) / 60000);
  const p = {
    participant_id: S.pid, condition: S.condition, task_id: S.task.id,
    task_completed: S.radios['grp-completed'] || 'Not answered',
    completion_time: document.getElementById('s_time').value || totalMin,
    q_answer_text: document.getElementById('s_answer').value,
    q_confidence: document.getElementById('s_confidence').value,
    q_toast_reliable: document.getElementById('toast_reliable').value,
    q_toast_confident: document.getElementById('toast_confident').value,
    q_toast_trustworthy: document.getElementById('toast_trustworthy').value,
    q_tlx_mental: document.getElementById('tlx_mental').value,
    q_tlx_effort: document.getElementById('tlx_effort').value,
    q_manipulation_check: S.radios['grp-manip'] || 'Not answered',
    q_verified: S.radios['grp-verified'] || 'Not answered',
    q_comments: document.getElementById('s_comments').value,
  };
  try {
    await fetch('/run/submit_survey', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [
        p.participant_id, p.condition, p.task_id, p.task_completed, p.completion_time,
        p.q_answer_text, p.q_confidence,
        p.q_toast_reliable, p.q_toast_confident, p.q_toast_trustworthy,
        p.q_tlx_mental, p.q_tlx_effort,
        p.q_manipulation_check, p.q_verified, p.q_comments,
        { participant_id: S.pid, condition: S.condition, query_count: S.queryCount, queries: S.queries }
      ]})
    });
  } catch(e) { console.warn('submit error:', e); }
  try {
    await fetch('/run/end_session', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [{ participant_id: S.pid, condition: S.condition, task_id: S.task.id, session_start: S.sessionStart / 1000, query_count: S.queryCount, queries: S.queries }] })
    });
  } catch(e) {}
  document.getElementById('done-pid').textContent   = S.pid;
  document.getElementById('done-cond').textContent  = S.condition;
  document.getElementById('done-task').textContent  = 'Task ' + S.task.id + ': ' + S.task.title;
  document.getElementById('done-queries').textContent = S.queryCount;
  document.getElementById('done-time').textContent  = totalMin + ' min';
  goTo(6);
}

// ── Logging ───────────────────────────────────────────────────────────
async function logToServer(eventType, data) {
  try {
    await fetch('/run/log_event_js', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [S.pid, S.condition, eventType, data] })
    });
  } catch(e) { console.log('[LOG]', eventType, data); }
}

// ── Researcher panel ──────────────────────────────────────────────────
function toggleResearcher() {
  const m = document.getElementById('researcher-modal');
  m.style.display = m.style.display === 'none' ? 'block' : 'none';
}

async function unlockResearcher() {
  const pw = document.getElementById('r_password').value;
  try {
    const resp = await fetch('/run/check_researcher_password', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [pw] })
    });
    const json = await resp.json();
    if (json.data[0] === true) {
      document.getElementById('researcher-locked').style.display = 'none';
      document.getElementById('researcher-open').style.display = 'block';
      loadStats();
      buildTasksPanel();
    } else {
      document.getElementById('r_error').textContent = '❌ Wrong password.';
    }
  } catch(e) { document.getElementById('r_error').textContent = 'Error: ' + e.message; }
}

function showRTab(name) {
  ['stats','tasks','log'].forEach(t => {
    document.getElementById('rpanel-' + t).style.display = t === name ? 'block' : 'none';
    document.getElementById('rtab-' + t).classList.toggle('active', t === name);
  });
  if (name === 'log') refreshLog();
  if (name === 'stats') loadStats();
}

async function loadStats() {
  try {
    const resp = await fetch('/run/get_researcher_stats', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [] })
    });
    const json = await resp.json();
    const d = json.data[0];
    document.getElementById('rs-total').textContent   = d.total_registered;
    document.getElementById('rs-a').textContent       = d.condition_a;
    document.getElementById('rs-b').textContent       = d.condition_b;
    document.getElementById('rs-surveys').textContent = d.surveys_submitted;
    const dist = document.getElementById('r-task-dist');
    dist.innerHTML = TASKS_DATA.map(t => `
      <div class="r-task-row">
        <span class="r-task-id">T${t.id}</span>
        <span class="r-task-name">${t.title}</span>
        <span class="r-task-count">${d.task_counts[t.id] || 0} sessions</span>
      </div>`).join('');
  } catch(e) { console.warn('stats error:', e); }
}

// Ground-truth answers stored in sessionStorage (researcher's browser only)
function buildTasksPanel() {
  const el = document.getElementById('r-tasks-list');
  el.innerHTML = TASKS_DATA.map(t => {
    const saved = sessionStorage.getItem('gt_' + t.id) || '';
    return `<div style="margin-bottom:1.25rem;padding-bottom:1.25rem;border-bottom:1px solid #e5e2da;">
      <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">
        <span style="font-family:'Courier New',monospace;font-size:0.72rem;background:#1a1917;color:#fff;padding:1px 6px;border-radius:3px;">T${t.id}</span>
        <span style="font-weight:bold;font-size:0.9rem;">${t.title}</span>
      </div>
      <div style="font-size:0.82rem;color:#6b6860;margin-bottom:8px;line-height:1.5;">${t.description}</div>
      <div class="r-gt-label">Ground-truth answer (researcher only — not shown to participants)</div>
      <textarea class="r-gt-input" rows="2" id="gt-${t.id}" placeholder="Fill in from your dataset…" onchange="sessionStorage.setItem('gt_${t.id}', this.value)">${saved}</textarea>
    </div>`;
  }).join('');
}

let _fullLog = '';
async function refreshLog() {
  document.getElementById('r_log_display').textContent = 'Loading…';
  try {
    const resp = await fetch('/run/load_log', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [] })
    });
    const json = await resp.json();
    _fullLog = json.data[0];
    filterLog();
  } catch(e) { document.getElementById('r_log_display').textContent = 'Error: ' + e.message; }
}

function filterLog() {
  const q = (document.getElementById('r-log-search').value || '').toLowerCase();
  if (!q) { document.getElementById('r_log_display').textContent = _fullLog; return; }
  const lines = _fullLog.split('\\n').filter(l => l.toLowerCase().includes(q));
  document.getElementById('r_log_display').textContent = lines.join('\\n') || '(no matches)';
}

function downloadLog() { window.open('/file=experiment_log.jsonl', '_blank'); }

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/\\n/g,'<br>');
}

init();
</script>
</body>
</html>
"""

# ── Gradio API endpoints ──────────────────────────────────────────────────────
def check_researcher_password(password):
    return password == RESEARCHER_PASSWORD

def log_event_js(participant_id, condition, event_type, data):
    log_event(participant_id, condition, event_type, data if isinstance(data, dict) else {})
    return True

# ── Build Gradio app ──────────────────────────────────────────────────────────
with gr.Blocks(title="Museum Collection Study") as demo:

    with gr.Row(visible=False):
        # generate_participant_id endpoint (NEW — returns P01, P02, …)
        _gpid_out = gr.Textbox()
        gr.Button("gpid").click(
            generate_participant_id, [], [_gpid_out],
            api_name="generate_participant_id"
        )

        # get_researcher_stats endpoint (NEW)
        _grs_out = gr.JSON()
        gr.Button("grs").click(
            get_researcher_stats, [], [_grs_out],
            api_name="get_researcher_stats"
        )

        # search endpoint
        _search_query  = gr.Textbox()
        _search_state  = gr.State({})
        _search_out    = gr.HTML()
        _search_state2 = gr.State({})
        gr.Button("search_api").click(
            search_condition_a,
            [_search_query, _search_state],
            [_search_out, _search_state2],
            api_name="search_condition_a"
        )

        # chat endpoint
        _chat_msg      = gr.Textbox()
        _chat_history  = gr.State([])
        _chat_state    = gr.State({})
        _chat_out      = gr.Textbox()
        _chat_history2 = gr.State([])
        _chat_state2   = gr.State({})
        _chat_history3 = gr.State([])
        gr.Button("chat_api").click(
            chat_condition_b,
            [_chat_msg, _chat_history, _chat_state],
            [_chat_out, _chat_history2, _chat_state2, _chat_history3],
            api_name="chat_condition_b"
        )

        # submit_survey endpoint
        _s_pid   = gr.Textbox()
        _s_cond  = gr.Textbox()
        _s_task  = gr.Number()
        _s_comp  = gr.Textbox()
        _s_time  = gr.Number()
        _s_ans   = gr.Textbox()
        _s_conf  = gr.Number()
        _s_tr    = gr.Number()
        _s_tc    = gr.Number()
        _s_tt    = gr.Number()
        _s_tlx1  = gr.Number()
        _s_tlx2  = gr.Number()
        _s_manip = gr.Textbox()
        _s_ver   = gr.Textbox()
        _s_com   = gr.Textbox()
        _s_state = gr.State({})
        _s_out   = gr.Textbox()
        def _submit_survey_wrap(pid, cond, task_id, completed, time_taken,
                                ans, conf, tr, tc, tt, tlx1, tlx2,
                                manip, ver, com, state):
            submit_survey(pid, cond, task_id, completed, time_taken,
                          ans, conf, tr, tc, tt, tlx1, tlx2, manip, ver, com, state)
            return "ok"
        gr.Button("submit_api").click(
            _submit_survey_wrap,
            [_s_pid, _s_cond, _s_task, _s_comp, _s_time,
             _s_ans, _s_conf, _s_tr, _s_tc, _s_tt,
             _s_tlx1, _s_tlx2, _s_manip, _s_ver, _s_com, _s_state],
            [_s_out],
            api_name="submit_survey"
        )

        # end_session endpoint
        _es_state = gr.State({})
        _es_out   = gr.Textbox()
        def _end_session_wrap(state):
            end_session(state)
            return "ok"
        gr.Button("end_api").click(
            _end_session_wrap, [_es_state], [_es_out],
            api_name="end_session"
        )

        # log_event_js endpoint
        _le_pid   = gr.Textbox()
        _le_cond  = gr.Textbox()
        _le_etype = gr.Textbox()
        _le_data  = gr.JSON()
        _le_out   = gr.Checkbox()
        gr.Button("log_api").click(
            log_event_js,
            [_le_pid, _le_cond, _le_etype, _le_data],
            [_le_out],
            api_name="log_event_js"
        )

        # researcher password check
        _rp_in  = gr.Textbox()
        _rp_out = gr.Checkbox()
        gr.Button("rp_api").click(
            check_researcher_password, [_rp_in], [_rp_out],
            api_name="check_researcher_password"
        )

        # load_log endpoint
        _ll_out = gr.Textbox()
        gr.Button("ll_api").click(
            load_log, [], [_ll_out],
            api_name="load_log"
        )

    gr.HTML(WIZARD_HTML)

demo.launch()