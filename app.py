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

TASKS = [
    {"id": 1, "title": "Find a ritual object",
     "description": "Find any ritual object in the collection. What is it called, when was it made, and what culture does it come from?"},
    {"id": 2, "title": "Identify the oldest item",
     "description": "What is the oldest item in the dataset? Provide its name, ID, and creation date."},
    {"id": 3, "title": "Find lacquerware",
     "description": "Find two examples of lacquerware in the collection. Who made them and when?"},
    {"id": 4, "title": "Artworks from 1700",
     "description": "How many items in the collection were created around 1700? List at least three."},
    {"id": 5, "title": "Anonymous creators",
     "description": "Find three items created by anonymous makers. What types of objects are they?"},
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

# ── Condition A: keyword search ───────────────────────────────────────────────
def keyword_search(query):
    import pandas as pd
    results = []
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="replace")
        q = query.lower()
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(q).any(), axis=1)
        for _, row in df[mask].head(8).iterrows():
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
    t0 = time.time()
    results = keyword_search(query)
    elapsed = round(time.time() - t0, 3)
    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"]     = state.get("queries", []) + [query]
    log_event(state.get("participant_id", "?"), "A", "search", {
        "task_id":      state.get("task_id"),
        "query":        query,
        "query_length": len(query),
        "query_number": state["query_count"],
        "result_count": len(results),
        "elapsed_s":    elapsed,
    })
    if not results:
        return "<p style='color:#e07b39;'>No results found. Try a different keyword.</p>", state
    cards = ""
    for r in results:
        img = (f"<img src='{r['image']}' style='width:80px;height:80px;object-fit:cover;"
               f"border-radius:6px;flex-shrink:0;' onerror=\"this.style.display='none'\">"
               if r["image"] and r["image"] != "nan" else "")
        link = (f"<a href='{r['url']}' target='_blank' style='color:#c77d3a;font-size:11px;'>View record</a>"
                if r["url"] and r["url"] != "nan" else "")
        cards += f"""
<div style='display:flex;gap:12px;align-items:flex-start;background:#f9f9f9;
            border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:10px;'>
  {img}
  <div>
    <div style='font-weight:600;font-size:15px;'>{r['title']}</div>
    <div style='color:#666;font-size:12px;margin-top:4px;'>
      Type: {r['type']} | Creator: {r['creator']} | Date: {r['date']}
    </div>
    <div style='margin-top:6px;'>{link}</div>
  </div>
</div>"""
    header = (f"<div style='font-size:12px;color:#888;margin-bottom:10px;'>"
              f"{len(results)} result(s) for \"{query}\" · {elapsed}s · keyword matches only</div>")
    return header + cards, state

# ── Condition B: RAG chat ─────────────────────────────────────────────────────
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
    return contents

def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history
    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"]     = state.get("queries", []) + [message]
    t0 = time.time()
    contents = _to_gemini_contents(history)
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT_B, tools=[file_search_tool]),
        )
        answer  = response.text or "No response generated."
        sources = []
        try:
            for candidate in response.candidates:
                if candidate.grounding_metadata:
                    for chunk in (candidate.grounding_metadata.grounding_chunks or []):
                        rc   = getattr(chunk, "retrieved_context", None)
                        name = rc and (getattr(rc, "title", None) or getattr(rc, "uri", None))
                        if name and name not in sources:
                            sources.append(name)
        except Exception:
            pass
        elapsed = round(time.time() - t0, 3)
        if any(p in answer.lower() for p in ["i'm not sure","i don't know","cannot find","not mentioned","no information"]):
            answer += "\n\n⚠️ Uncertainty notice: The AI indicated limited confidence. Please verify with the original source."
        if sources:
            answer += "\n\nSources: " + " · ".join(s.replace('.txt','').replace('.pdf','') for s in sources)
        log_event(state.get("participant_id","?"), "B", "chat", {
            "task_id":         state.get("task_id"),
            "query":           message,
            "query_length":    len(message),
            "query_number":    state["query_count"],
            "response_text":   answer,
            "response_length": len(answer),
            "sources":         sources,
            "elapsed_s":       elapsed,
        })
    except Exception as e:
        answer = f"Error: {e}"
    new_history = list(history or []) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", new_history, state, new_history

# ── Pre-survey ────────────────────────────────────────────────────────────────
def submit_pre_survey(pid, age, edu, lang, museum, ai_use, a1, a2, a3, a4, search_c, state):
    log_event(pid, "?", "pre_survey", {
        "participant_id": pid, "age": age, "education": edu,
        "native_language": lang, "museum_familiarity": museum, "ai_usage_freq": ai_use,
        "aias4_item1": a1, "aias4_item2": a2, "aias4_item3": a3, "aias4_item4": a4,
        "search_comfort": search_c,
    })

# ── Final survey ──────────────────────────────────────────────────────────────
def submit_final_survey(toast_r, toast_c, toast_t, tlx_m, tlx_e,
                        sus1, sus2, sus3, sus4, sus5,
                        verified, manip, comments, state):
    def i(v):
        try: return int(v)
        except: return v
    log_event(state.get("participant_id","?"), state.get("condition","?"), "final_survey", {
        "toast_reliable":         i(toast_r), "toast_confident": i(toast_c), "toast_trustworthy": i(toast_t),
        "tlx_mental_demand":      i(tlx_m),   "tlx_effort":       i(tlx_e),
        "sus_easy_to_use":        i(sus1),    "sus_confident":    i(sus2),
        "sus_would_reuse":        i(sus3),    "sus_reliable_info":i(sus4),
        "sus_understood_sources": i(sus5),
        "verified_sources":       verified,   "manipulation_check": manip,
        "comments":               comments,
        "task_results":           state.get("task_results", []),
        "total_queries":          state.get("total_query_count", 0),
        "total_time_s":           round(time.time() - state.get("session_start", time.time()), 1),
    })

def load_log():
    try:
        with open(LOG_FILE) as f: return f.read()
    except FileNotFoundError:
        return "No log entries yet."

def download_log():
    return LOG_FILE if os.path.exists(LOG_FILE) else None

# ── UI helpers ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.stepper{display:flex;gap:0;margin-bottom:28px;font-size:13px;font-weight:500;}
.step{flex:1;text-align:center;padding:10px 4px;background:#e2e8f0;color:#64748b;border-right:2px solid white;}
.step:first-child{border-radius:8px 0 0 8px;}
.step:last-child{border-radius:0 8px 8px 0;border-right:none;}
.step.active{background:#3b82f6;color:white;font-weight:700;}
.step.done{background:#bbf7d0;color:#166534;}
.task-card{background:#eff6ff;border-left:4px solid #3b82f6;padding:14px 18px;border-radius:8px;margin-bottom:20px;font-size:15px;}
.done-screen{text-align:center;padding:60px 20px;}
footer{display:none !important;}
"""

EMPTY_RESULTS = "<p style='color:#888;padding:20px;'>Results will appear here.</p>"

def make_progress(step, task_index=None):
    labels = ["1 Pre-survey", "2 Setup", "3 Tasks", "4 Survey", "5 Done"]
    if step == 3 and task_index is not None:
        labels[2] = f"3 Task {task_index+1}/5"
    parts = []
    for i, label in enumerate(labels, 1):
        css = "step active" if i == step else ("step done" if i < step else "step")
        parts.append(f'<div class="{css}">{label}</div>')
    return f'<div class="stepper">{"".join(parts)}</div>'

def task_card_html(idx):
    t    = TASKS[idx]
    dots = "● " * (idx + 1) + "○ " * (4 - idx)
    return (f"<div class='task-card'>"
            f"<span style='font-size:20px;letter-spacing:2px;'>{dots.strip()}</span><br>"
            f"<strong style='font-size:16px;'>Task {t['id']}/5 — {t['title']}</strong><br>"
            f"<span style='color:#334155;'>{t['description']}</span></div>")

def mini_header_html(idx):
    t = TASKS[idx]
    return (f"<div style='background:#f0fdf4;border-left:4px solid #22c55e;"
            f"padding:14px 18px;border-radius:8px;margin-bottom:16px;'>"
            f"<strong>📝 Task {t['id']}/5 complete — Quick check</strong><br>"
            f"<span style='color:#64748b;font-size:14px;'>{t['description']}</span></div>")

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Museum Collection Study",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
    css=CUSTOM_CSS,
) as demo:

    session_state = gr.State({})
    chat_history  = gr.State([])

    gr.Markdown("# Museum Collection Study")
    progress_bar = gr.HTML(make_progress(1))

    # ── Step 1: Pre-survey ────────────────────────────────────────────────────
    with gr.Column(visible=True) as step1_col:
        gr.Markdown("## Step 1 — Before you begin\nAll responses are anonymous and will only be used for research.")
        pre_pid = gr.Textbox(label="Participant ID (e.g. P01)", placeholder="P01")

        gr.Markdown("#### 👤 Demographics")
        with gr.Row():
            pre_age  = gr.Number(label="Age", minimum=18, maximum=99, value=25)
            pre_edu  = gr.Dropdown(label="Education level",
                choices=["Secondary / high school","Bachelor's degree","Master's degree","PhD / doctorate","Other"])
            pre_lang = gr.Textbox(label="Native language", placeholder="e.g. Dutch")

        gr.Markdown("#### 🛠️ Tool familiarity")
        pre_museum = gr.Slider(1, 5, step=1, value=3, label="Museum / art history familiarity",
                               info="1 = no knowledge  ·  5 = expert")
        pre_ai     = gr.Dropdown(label="How often do you use AI tools like ChatGPT?",
            choices=["Never","Rarely (few times/year)","Sometimes (monthly)","Often (weekly)","Daily"])
        pre_search = gr.Slider(1, 5, step=1, value=3, label="Comfort searching databases / catalogues",
                               info="1 = not at all  ·  5 = very comfortable")

        gr.Markdown("#### 🤖 Attitude towards AI (AIAS-4)\n*1 = strongly disagree · 5 = strongly agree*")
        with gr.Row():
            pre_a1 = gr.Slider(1,5,step=1,value=3, label="AI systems perform as well as humans")
            pre_a2 = gr.Slider(1,5,step=1,value=3, label="I'm comfortable relying on AI for information")
        with gr.Row():
            pre_a3 = gr.Slider(1,5,step=1,value=3, label="AI tools are useful for everyday work")
            pre_a4 = gr.Slider(1,5,step=1,value=3, label="I trust AI-generated results to be accurate")

        pre_btn = gr.Button("Save & continue →", variant="primary", size="lg")
        pre_out = gr.Markdown("")

    # ── Step 2: Setup ─────────────────────────────────────────────────────────
    with gr.Column(visible=False) as step2_col:
        gr.Markdown("## Step 2 — Session Setup\n*Filled in by the researcher.*")
        with gr.Row():
            pid_box  = gr.Textbox(label="Participant ID", placeholder="P01")
            cond_box = gr.Dropdown(label="Condition",
                choices=["A — Keyword Search","B — AI Chat"], value="A — Keyword Search")
        gr.HTML("<div style='background:#fefce8;border:1px solid #fbbf24;border-radius:8px;"
                "padding:12px 16px;margin:8px 0;font-size:14px;'>"
                "ℹ️ The participant will complete <strong>all 5 tasks</strong> in sequence "
                "in their assigned condition.</div>")
        setup_btn = gr.Button("Start session →", variant="primary", size="lg")
        setup_out = gr.Markdown("")

    # ── Step 3: Task screen ───────────────────────────────────────────────────
    with gr.Column(visible=False) as task_col:
        task_card = gr.HTML(task_card_html(0))

        with gr.Column(visible=True) as cond_a_col:
            gr.Markdown("### 🔍 Keyword Search\n*Results are direct matches — no AI interpretation.*")
            with gr.Row():
                search_box = gr.Textbox(placeholder="e.g. ritual object, 1700, lacquer...",
                                        show_label=False, scale=8)
                search_btn = gr.Button("Search", variant="primary", scale=1)
            search_results = gr.HTML(EMPTY_RESULTS)

        with gr.Column(visible=False) as cond_b_col:
            gr.Markdown("### 🤖 AI Research Assistant\n*AI answers may contain errors — always verify with source links.*")
            chatbot  = gr.Chatbot(label="Chat", height=400)
            with gr.Row():
                chat_box = gr.Textbox(placeholder="Ask about the collection...",
                                      show_label=False, scale=9)
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat", size="sm")

        gr.Markdown("---")
        finish_btn = gr.Button("✅ Finish task →", variant="secondary", size="lg")
        finish_out = gr.Markdown("")

    # ── Step 3b: Mini-survey (per task) ───────────────────────────────────────
    with gr.Column(visible=False) as mini_col:
        mini_header = gr.HTML(mini_header_html(0))

        mini_answer = gr.Textbox(
            label="📝 What is your answer to this task?",
            lines=2, placeholder="Write your answer here...")

        mini_completed = gr.Radio(
            label="✅ Did you complete the task?",
            choices=["✅ Yes", "⚠️ Partially", "❌ No"])

        gr.Markdown("**📊 How confident are you in your answer?**  \n"
                    "*1 = not at all confident · 7 = completely sure*")
        mini_confidence = gr.Radio(
            choices=["1","2","3","4","5","6","7"],
            label="", show_label=False)

        mini_btn = gr.Button("Continue →", variant="primary", size="lg")
        mini_out = gr.Markdown("")

    # ── Step 4: Final survey ──────────────────────────────────────────────────
    with gr.Column(visible=False) as final_col:
        gr.Markdown("## Step 4 — Final survey\n"
                    "You've completed all 5 tasks! Please answer a few last questions about your overall experience.")
        survey_summary = gr.Markdown("")

        with gr.Accordion("🤝 Trust in the system (TOAST)", open=True):
            gr.Markdown("*Rate from 1 (strongly disagree) to 7 (strongly agree)*")
            s_toast_r = gr.Radio(["1","2","3","4","5","6","7"],
                                 label="The system performed reliably")
            s_toast_c = gr.Radio(["1","2","3","4","5","6","7"],
                                 label="I felt confident using this system")
            s_toast_t = gr.Radio(["1","2","3","4","5","6","7"],
                                 label="I found the system trustworthy")

        with gr.Accordion("🧠 Mental effort — NASA-TLX", open=True):
            gr.Markdown("*Rate from 1 (not at all) to 7 (extremely)*")
            s_tlx_m = gr.Radio(["1","2","3","4","5","6","7"],
                               label="How mentally demanding was the overall session?")
            s_tlx_e = gr.Radio(["1","2","3","4","5","6","7"],
                               label="How hard did you have to work overall?")

        with gr.Accordion("💻 Usability", open=True):
            gr.Markdown("*Rate from 1 (strongly disagree) to 5 (strongly agree)*")
            s_sus1 = gr.Radio(["1","2","3","4","5"], label="I found the system easy to use")
            s_sus2 = gr.Radio(["1","2","3","4","5"], label="I felt confident using the system")
            s_sus3 = gr.Radio(["1","2","3","4","5"], label="I would use this system again")
            s_sus4 = gr.Radio(["1","2","3","4","5"], label="The system gave me reliable information")
            s_sus5 = gr.Radio(["1","2","3","4","5"], label="I understood where the answers came from")

        with gr.Accordion("🔍 Critical evaluation", open=True):
            s_verified = gr.Radio(
                label="Did you verify any answers using the source link?",
                choices=["✅ Yes, always","🔁 Sometimes","❌ No"])
            s_manip = gr.Radio(
                label="What kind of tool did you use during this session?",
                choices=["Keyword search","AI assistant","Both","Not sure"])
            s_comments = gr.Textbox(
                label="💬 Any comments or feedback? (optional)", lines=3,
                placeholder="What worked well? What was frustrating?")

        final_btn = gr.Button("Submit survey →", variant="primary", size="lg")
        final_out = gr.Markdown("")

    # ── Step 5: Done ──────────────────────────────────────────────────────────
    with gr.Column(visible=False) as done_col:
        gr.HTML("""
        <div class='done-screen'>
          <div style='font-size:64px;margin-bottom:16px;'>✅</div>
          <h2 style='font-size:26px;color:#166534;margin-bottom:8px;'>Thank you for participating!</h2>
          <p style='color:#64748b;font-size:16px;'>The researcher will be with you shortly.</p>
        </div>
        """)

    # ── Researcher view ───────────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Accordion("🔒 Researcher View", open=False):
        r_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
        r_unlock   = gr.Button("Unlock", variant="primary")
        r_msg      = gr.Markdown("")
        r_log      = gr.Code(label="experiment_log.jsonl", language="json", lines=30, visible=False)
        with gr.Row(visible=False) as r_btn_row:
            r_refresh  = gr.Button("Refresh", variant="secondary")
            r_download = gr.Button("⬇️ Download log", variant="primary")
        r_file = gr.File(label="Download", visible=False)

    # ── Event handlers ─────────────────────────────────────────────────────────

    # Step 1 → 2
    def on_pre_submit(pid, age, edu, lang, museum, ai_use, a1, a2, a3, a4, search_c, state):
        if not pid.strip():
            return ("⚠️ Please enter a Participant ID.",
                    gr.update(), gr.update(visible=True), gr.update(visible=False))
        submit_pre_survey(pid, age, edu, lang, museum, ai_use, a1, a2, a3, a4, search_c, state)
        return (f"✅ Saved for **{pid}**.",
                gr.update(value=make_progress(2)),
                gr.update(visible=False),
                gr.update(visible=True))

    pre_btn.click(
        on_pre_submit,
        [pre_pid, pre_age, pre_edu, pre_lang, pre_museum, pre_ai,
         pre_a1, pre_a2, pre_a3, pre_a4, pre_search, session_state],
        [pre_out, progress_bar, step1_col, step2_col],
    )

    # Step 2 → Task 1
    def on_start_session(pid, cond, state):
        if not pid.strip():
            return ("⚠️ Please enter a Participant ID.",
                    state, gr.update(), gr.update(visible=False), gr.update(visible=True),
                    gr.update(), gr.update(visible=True), gr.update(visible=False))
        state.update({
            "participant_id":    pid,
            "condition":         cond,
            "session_start":     time.time(),
            "task_index":        0,
            "task_id":           TASKS[0]["id"],
            "task_start_time":   time.time(),
            "query_count":       0,
            "queries":           [],
            "task_results":      [],
            "total_query_count": 0,
        })
        log_event(pid, cond, "session_start", {"condition": cond})
        log_event(pid, cond, "task_start",    {"task_id": 1, "task_index": 0})
        is_a = "A" in cond
        return (
            f"✅ Session started — {pid} · {'Condition A (Keyword Search)' if is_a else 'Condition B (AI Chat)'}",
            state,
            gr.update(value=make_progress(3, 0)),
            gr.update(visible=False),        # step2_col
            gr.update(visible=True),         # task_col
            gr.update(value=task_card_html(0)),
            gr.update(visible=is_a),         # cond_a_col
            gr.update(visible=not is_a),     # cond_b_col
        )

    setup_btn.click(
        on_start_session,
        [pid_box, cond_box, session_state],
        [setup_out, session_state, progress_bar,
         step2_col, task_col, task_card, cond_a_col, cond_b_col],
    )

    # Finish task → mini-survey
    def on_finish_task(state):
        idx     = state.get("task_index", 0)
        elapsed = round(time.time() - state.get("task_start_time", time.time()), 1)
        state["last_task_elapsed"] = elapsed
        log_event(state.get("participant_id","?"), state.get("condition","?"), "task_end", {
            "task_id":     TASKS[idx]["id"],
            "task_index":  idx,
            "elapsed_s":   elapsed,
            "query_count": state.get("query_count", 0),
            "queries":     state.get("queries", []),
        })
        return (
            "",
            state,
            gr.update(value=make_progress(3, idx)),
            gr.update(visible=False),                    # task_col
            gr.update(visible=True),                     # mini_col
            gr.update(value=mini_header_html(idx)),
        )

    finish_btn.click(
        on_finish_task,
        [session_state],
        [finish_out, session_state, progress_bar, task_col, mini_col, mini_header],
    )

    # Mini-survey submit → next task or final survey
    def on_mini_submit(answer, completed, confidence, state, history):
        idx  = state.get("task_index", 0)
        pid  = state.get("participant_id", "?")
        cond = state.get("condition", "?")

        result = {
            "task_id":     TASKS[idx]["id"],
            "task_index":  idx,
            "answer":      answer,
            "completed":   completed,
            "confidence":  int(confidence) if confidence else None,
            "elapsed_s":   state.get("last_task_elapsed", 0),
            "query_count": state.get("query_count", 0),
            "queries":     state.get("queries", []),
        }
        state["task_results"]      = state.get("task_results", []) + [result]
        state["total_query_count"] = state.get("total_query_count", 0) + state.get("query_count", 0)

        log_event(pid, cond, "task_survey", {
            "task_id":    TASKS[idx]["id"],
            "answer":     answer,
            "completed":  completed,
            "confidence": result["confidence"],
        })

        if idx < 4:
            # Advance to next task
            next_idx = idx + 1
            state.update({
                "task_index":      next_idx,
                "task_id":         TASKS[next_idx]["id"],
                "task_start_time": time.time(),
                "query_count":     0,
                "queries":         [],
            })
            log_event(pid, cond, "task_start", {"task_id": TASKS[next_idx]["id"], "task_index": next_idx})
            return (
                "",
                state, [],
                gr.update(value=make_progress(3, next_idx)),
                gr.update(visible=False),                          # mini_col
                gr.update(visible=True),                           # task_col
                gr.update(visible=False),                          # final_col
                gr.update(value=task_card_html(next_idx)),
                gr.update(value=""),                               # search_box reset
                gr.update(value=EMPTY_RESULTS),                    # search_results reset
                gr.update(value=[]),                               # chatbot reset
                gr.update(),                                       # survey_summary (no change)
            )
        else:
            # All 5 tasks done → show final survey
            log_event(pid, cond, "all_tasks_complete", {
                "total_queries": state.get("total_query_count", 0)
            })
            rows = "\n".join(
                f"| Task {r['task_id']} | {r.get('completed','—')} | "
                f"Confidence: {r.get('confidence','—')}/7 | "
                f"{r.get('query_count',0)} queries | {r.get('elapsed_s',0)}s |"
                for r in state.get("task_results", [])
            )
            summary = (
                "**Your session at a glance:**\n\n"
                "| Task | Status | Confidence | Queries | Time |\n"
                "|------|--------|------------|---------|------|\n"
                + rows
            )
            return (
                "",
                state, history,
                gr.update(value=make_progress(4)),
                gr.update(visible=False),   # mini_col
                gr.update(visible=False),   # task_col
                gr.update(visible=True),    # final_col
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=summary),   # survey_summary
            )

    mini_btn.click(
        on_mini_submit,
        [mini_answer, mini_completed, mini_confidence, session_state, chat_history],
        [mini_out, session_state, chat_history, progress_bar,
         mini_col, task_col, final_col,
         task_card, search_box, search_results, chatbot, survey_summary],
    )

    # Final survey → done
    def on_final_submit(toast_r, toast_c, toast_t, tlx_m, tlx_e,
                        sus1, sus2, sus3, sus4, sus5,
                        verified, manip, comments, state):
        submit_final_survey(toast_r, toast_c, toast_t, tlx_m, tlx_e,
                            sus1, sus2, sus3, sus4, sus5,
                            verified, manip, comments, state)
        return (
            "✅ Survey submitted. Thank you!",
            gr.update(value=make_progress(5)),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    final_btn.click(
        on_final_submit,
        [s_toast_r, s_toast_c, s_toast_t, s_tlx_m, s_tlx_e,
         s_sus1, s_sus2, s_sus3, s_sus4, s_sus5,
         s_verified, s_manip, s_comments, session_state],
        [final_out, progress_bar, final_col, done_col],
    )

    # Condition A
    search_btn.click(search_condition_a, [search_box, session_state], [search_results, session_state])
    search_box.submit(search_condition_a, [search_box, session_state], [search_results, session_state])

    # Condition B
    chat_btn.click(chat_condition_b,
                   [chat_box, chat_history, session_state],
                   [chat_box, chatbot, session_state, chat_history])
    chat_box.submit(chat_condition_b,
                    [chat_box, chat_history, session_state],
                    [chat_box, chatbot, session_state, chat_history])
    clear_btn.click(lambda: ([], []), None, [chatbot, chat_history])

    # Researcher view
    def unlock(pw):
        if pw == RESEARCHER_PASSWORD:
            return ("✅ Access granted.",
                    gr.update(visible=True, value=load_log()),
                    gr.update(visible=True),
                    gr.update(visible=True))
        return ("❌ Wrong password.",
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

    r_unlock.click(unlock, [r_password], [r_msg, r_log, r_btn_row, r_file])
    r_refresh.click(load_log, None, r_log)
    r_download.click(download_log, None, r_file)

demo.launch()
