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
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=API_KEY)

file_search_tool = types.Tool(
    file_search=types.FileSearch(file_search_store_names=[STORE_NAME])
)

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
            img_html = f"<img src='{r['image']}' style='width:80px;height:80px;object-fit:cover;border-radius:6px;flex-shrink:0;' onerror=\"this.style.display='none'\">"
        link = f"<a href='{r['url']}' target='_blank' style='color:#c77d3a;font-size:11px;'>View record</a>" if r["url"] and r["url"] != "nan" else ""
        cards += f"""
<div style='display:flex;gap:12px;align-items:flex-start;background:#f9f9f9;border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:10px;'>
    {img_html}
    <div>
        <div style='font-weight:600;font-size:15px;'>{r['title']}</div>
        <div style='color:#666;font-size:12px;margin-top:4px;'>
            Type: {r['type']} | Creator: {r['creator']} | Date: {r['date']}
        </div>
        <div style='margin-top:6px;'>{link}</div>
    </div>
</div>"""

    html = f"<div style='font-size:12px;color:#888;margin-bottom:10px;'>{len(results)} result(s) for \"{query}\" · {elapsed}s · Keyword matches only, no AI interpretation.</div>{cards}"
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
            src_links = " · ".join(s.replace('.txt', '').replace('.pdf', '') for s in sources)
            answer += f"\n\nSources: {src_links}"

        log_event(state.get("participant_id", "unknown"), "B", "chat", {
            "query":           message,
            "query_length":    len(message),
            "query_number":    state.get("query_count", 0),
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

# ── Pre-task survey ───────────────────────────────────────────────────────────
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
    return f"✅ Pre-task survey saved for **{pid}**."

# ── Post-task survey ──────────────────────────────────────────────────────────
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
    return (
        f"Survey submitted! Thank you, {pid}.\n\n"
        f"Condition: {condition} · Task: {task_id} · Queries made: {state.get('query_count', 0)}\n\n"
        f"Your responses have been saved."
    )

# ── Session end ───────────────────────────────────────────────────────────────
def end_session(state):
    elapsed = round(time.time() - state.get("session_start", time.time()), 1)
    log_event(state.get("participant_id", "unknown"), state.get("condition", "?"), "session_end", {
        "task_id":       state.get("task_id"),
        "total_time_s":  elapsed,
        "total_queries": state.get("query_count", 0),
        "all_queries":   state.get("queries", []),
    })
    return f"✅ Session ended. Total time: {elapsed}s · Total queries: {state.get('query_count', 0)}"

# ── Log helpers ───────────────────────────────────────────────────────────────
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

# ── CSS & progress bar ────────────────────────────────────────────────────────
CUSTOM_CSS = """
.stepper { display:flex; gap:0; margin-bottom:28px; font-size:13px; font-weight:500; }
.step { flex:1; text-align:center; padding:10px 4px; background:#e2e8f0; color:#64748b; border-right:2px solid white; }
.step:first-child { border-radius:8px 0 0 8px; }
.step:last-child  { border-radius:0 8px 8px 0; border-right:none; }
.step.active { background:#3b82f6; color:white; font-weight:700; }
.step.done   { background:#bbf7d0; color:#166534; }
.task-card { background:#eff6ff; border-left:4px solid #3b82f6; padding:14px 18px; border-radius:8px; margin-bottom:20px; font-size:15px; }
.done-screen { text-align:center; padding:60px 20px; }
footer { display:none !important; }
"""

def make_progress(step):
    labels = ["1 Pre-survey", "2 Setup", "3 Task", "4 Survey", "5 Done"]
    parts = []
    for i, label in enumerate(labels, 1):
        css = "step active" if i == step else ("step done" if i < step else "step")
        parts.append(f'<div class="{css}">{label}</div>')
    return f'<div class="stepper">{"".join(parts)}</div>'

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Museum Collection Study",
    theme=gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
) as demo:

    session_state = gr.State({})
    chat_history  = gr.State([])

    gr.Markdown("# Museum Collection Study")
    progress_bar = gr.HTML(make_progress(1))

    # ── Step 1: Pre-survey ────────────────────────────────────────────────────
    with gr.Column(visible=True) as step1_col:
        gr.Markdown(
            "## Step 1 — Before you begin\n"
            "Please answer these short questions. All responses are anonymous."
        )

        pre_pid = gr.Textbox(label="Participant ID (e.g. P01)", placeholder="P01")

        gr.Markdown("#### Demographics")
        with gr.Row():
            pre_age       = gr.Number(label="Age", minimum=18, maximum=99, value=25)
            pre_education = gr.Dropdown(
                label="Highest education level",
                choices=["Secondary / high school", "Bachelor's degree", "Master's degree",
                         "PhD / doctorate", "Other"],
            )
            pre_language  = gr.Textbox(label="Native language", placeholder="e.g. Dutch")

        gr.Markdown("#### Familiarity with tools")
        pre_museum = gr.Slider(1, 5, step=1, value=3,
                               label="Museum / art history familiarity",
                               info="1 = no knowledge  ·  5 = expert")
        pre_ai_usage = gr.Dropdown(
            label="How often do you use ChatGPT-style AI tools?",
            choices=["Never", "Rarely (a few times a year)", "Sometimes (monthly)",
                     "Often (weekly)", "Daily"],
        )
        pre_search = gr.Slider(1, 5, step=1, value=3,
                               label="Comfort searching databases / library catalogues",
                               info="1 = not at all  ·  5 = very comfortable")

        gr.Markdown(
            "#### Attitude towards AI (AIAS-4)\n"
            "Rate each statement from 1 (strongly disagree) to 5 (strongly agree)."
        )
        with gr.Row():
            pre_aias1 = gr.Slider(1, 5, step=1, value=3,
                                  label="AI systems perform tasks as well as humans",
                                  info="1 = strongly disagree  ·  5 = strongly agree")
            pre_aias2 = gr.Slider(1, 5, step=1, value=3,
                                  label="I feel comfortable relying on AI for information",
                                  info="1 = strongly disagree  ·  5 = strongly agree")
        with gr.Row():
            pre_aias3 = gr.Slider(1, 5, step=1, value=3,
                                  label="AI tools are a useful addition to everyday work",
                                  info="1 = strongly disagree  ·  5 = strongly agree")
            pre_aias4 = gr.Slider(1, 5, step=1, value=3,
                                  label="I trust AI-generated results to be mostly accurate",
                                  info="1 = strongly disagree  ·  5 = strongly agree")

        pre_submit_btn = gr.Button("Save & continue →", variant="primary", size="lg")
        pre_submit_out = gr.Markdown("")

    # ── Step 2: Setup (researcher) ────────────────────────────────────────────
    with gr.Column(visible=False) as step2_col:
        gr.Markdown(
            "## Step 2 — Session Setup\n"
            "*Filled in by the researcher before the participant starts.*"
        )

        with gr.Row():
            pid_box  = gr.Textbox(label="Participant ID (e.g. P01)", placeholder="P01")
            cond_box = gr.Dropdown(
                label="Condition (assigned by researcher)",
                choices=["A — Keyword Search", "B — AI Chat"],
                value="A — Keyword Search",
            )

        task_dropdown = gr.Dropdown(
            label="Select task",
            choices=[f"Task {t['id']}: {t['title']}" for t in TASKS],
            value="Task 1: Find a ritual object",
        )

        setup_btn = gr.Button("Start session →", variant="primary", size="lg")
        setup_out = gr.Markdown("")

    # ── Step 3: Task ──────────────────────────────────────────────────────────
    with gr.Column(visible=False) as step3_col:
        gr.Markdown("## Step 3 — Complete your task")
        task_card = gr.HTML(
            "<div class='task-card'><strong>Your task:</strong> will appear here after setup.</div>"
        )

        with gr.Column(visible=True) as cond_a_col:
            gr.Markdown(
                "### Keyword Search\n"
                "Type keywords and press Enter or click Search. "
                "Results are direct matches — no AI interpretation."
            )
            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="e.g. ritual object, 1700, lacquer...",
                    show_label=False, scale=8
                )
                search_btn = gr.Button("Search", variant="primary", scale=1)
            search_results = gr.HTML(
                "<p style='color:#888;padding:20px;'>Results will appear here.</p>"
            )

        with gr.Column(visible=False) as cond_b_col:
            gr.Markdown(
                "### AI Research Assistant\n"
                "Ask questions in natural language. "
                "AI answers may contain errors — always verify with source links."
            )
            chatbot = gr.Chatbot(label="Chat", height=420)
            with gr.Row():
                chat_box = gr.Textbox(
                    placeholder="Ask about the collection...",
                    show_label=False, scale=9
                )
                chat_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat", size="sm")

        gr.Markdown("---")
        finish_btn = gr.Button("Finish task →", variant="secondary", size="lg")
        finish_out = gr.Markdown("")

    # ── Step 4: Post-survey ───────────────────────────────────────────────────
    with gr.Column(visible=False) as step4_col:
        gr.Markdown(
            "## Step 4 — Post-task survey\n"
            "Complete this after finishing your task.\n\n"
            "Sliders: 1 = strongly disagree / not at all · 7 = strongly agree / extremely."
        )

        survey_session_info = gr.Markdown(
            "*Session info will be filled in automatically.*"
        )

        with gr.Row():
            s_completed = gr.Radio(
                label="Did you complete the task?",
                choices=["Yes", "Partially", "No"]
            )
            s_time = gr.Number(label="Approx. time taken (minutes)", value=5)

        gr.Markdown("#### Your answer (H2 — Accuracy)")
        s_answer_text = gr.Textbox(
            label="What is your final answer to the task? (write it out fully)",
            lines=3,
            placeholder="e.g. The ritual object is called X, made in year Y, from culture Z."
        )
        s_confidence = gr.Slider(1, 7, step=1, value=4,
                                 label="How confident are you that your answer is correct?",
                                 info="1 = not at all confident  ·  7 = completely sure")

        gr.Markdown("#### TOAST Trust Scale (H3)")
        with gr.Row():
            s_toast_reliable    = gr.Slider(1, 7, step=1, value=4,
                                            label="The system performed reliably",
                                            info="1 = strongly disagree  ·  7 = strongly agree")
            s_toast_confident   = gr.Slider(1, 7, step=1, value=4,
                                            label="I felt confident using this system",
                                            info="1 = strongly disagree  ·  7 = strongly agree")
            s_toast_trustworthy = gr.Slider(1, 7, step=1, value=4,
                                            label="I found the system trustworthy",
                                            info="1 = strongly disagree  ·  7 = strongly agree")

        gr.Markdown("#### Cognitive Load — NASA-TLX (H4)")
        with gr.Row():
            s_tlx_mental = gr.Slider(1, 7, step=1, value=4,
                                     label="How mentally demanding was the task?",
                                     info="1 = not at all  ·  7 = extremely")
            s_tlx_effort = gr.Slider(1, 7, step=1, value=4,
                                     label="How hard did you have to work?",
                                     info="1 = very little  ·  7 = very hard")

        gr.Markdown("#### Manipulation check")
        s_manipulation = gr.Radio(
            label="What kind of tool did you just use?",
            choices=["Keyword search", "AI assistant", "Both", "Not sure"]
        )

        gr.Markdown("#### Critical evaluation")
        s_verified = gr.Radio(
            label="Did you verify any answers with the source link?",
            choices=["Yes, always", "Sometimes", "No"]
        )
        s_comments = gr.Textbox(
            label="Any comments or feedback? (optional)",
            lines=3,
            placeholder="What worked well? What was frustrating?"
        )

        survey_btn = gr.Button("Submit survey →", variant="primary", size="lg")
        survey_out = gr.Markdown("")

    # ── Step 5: Done ──────────────────────────────────────────────────────────
    with gr.Column(visible=False) as step5_col:
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
        gr.Markdown("*This section is for the researcher only. Participants do not need this.*")
        researcher_password = gr.Textbox(
            label="Researcher password", type="password",
            placeholder="Enter password to unlock"
        )
        unlock_btn  = gr.Button("Unlock", variant="primary")
        access_msg  = gr.Markdown("")
        log_display = gr.Code(label="experiment_log.jsonl", language="json", lines=30, visible=False)
        with gr.Row(visible=False) as button_row:
            refresh_btn  = gr.Button("Refresh log", variant="secondary")
            download_btn = gr.Button("⬇️ Download log", variant="primary")
        download_file = gr.File(label="Download", visible=False)

    # ── Event handlers ─────────────────────────────────────────────────────────

    # Step 1 → 2: save pre-survey, show step 2
    def on_pre_submit(pid, age, education, language, museum, ai_usage,
                      aias1, aias2, aias3, aias4, search_comfort, state):
        if not pid.strip():
            return (
                "⚠️ Please enter a Participant ID before continuing.",
                gr.update(), gr.update(visible=True), gr.update(visible=False),
            )
        msg = submit_pre_survey(pid, age, education, language, museum, ai_usage,
                                aias1, aias2, aias3, aias4, search_comfort, state)
        return (
            msg,
            gr.update(value=make_progress(2)),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    pre_submit_btn.click(
        on_pre_submit,
        [pre_pid, pre_age, pre_education, pre_language, pre_museum, pre_ai_usage,
         pre_aias1, pre_aias2, pre_aias3, pre_aias4, pre_search, session_state],
        [pre_submit_out, progress_bar, step1_col, step2_col],
    )

    # Step 2 → 3: start session, show correct condition pane
    def on_start_session(pid, cond, task_label, state):
        if not pid.strip():
            return (
                "⚠️ Please enter a Participant ID.",
                state, gr.update(), gr.update(visible=False), gr.update(visible=True),
                gr.update(), gr.update(visible=True), gr.update(visible=False),
            )
        task_id = int(task_label.split(":")[0].replace("Task ", "").strip())
        task = next(t for t in TASKS if t["id"] == task_id)
        state["participant_id"] = pid
        state["condition"]      = cond
        state["task_id"]        = task_id
        state["session_start"]  = time.time()
        state["query_count"]    = 0
        state["queries"]        = []
        log_event(pid, cond, "session_start", {"task_id": task_id})

        task_html = (
            f"<div class='task-card'>"
            f"<strong>Task {task_id}: {task['title']}</strong><br>"
            f"{task['description']}"
            f"</div>"
        )
        is_a = "A" in cond

        return (
            f"✅ Session started — {pid} · {'Condition A' if is_a else 'Condition B'} · Task {task_id}",
            state,
            gr.update(value=make_progress(3)),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=task_html),
            gr.update(visible=is_a),
            gr.update(visible=not is_a),
        )

    setup_btn.click(
        on_start_session,
        [pid_box, cond_box, task_dropdown, session_state],
        [setup_out, session_state, progress_bar,
         step2_col, step3_col, task_card, cond_a_col, cond_b_col],
    )

    # Step 3 → 4: end session, auto-fill survey header
    def on_finish_task(state):
        msg  = end_session(state)
        pid  = state.get("participant_id", "?")
        cond = state.get("condition", "?")
        tid  = state.get("task_id", "?")
        info = (
            f"**Participant:** {pid}  ·  **Condition:** {cond}  ·  **Task:** {tid}  ·  "
            f"**Queries made:** {state.get('query_count', 0)}"
        )
        return (
            msg,
            gr.update(value=make_progress(4)),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=info),
        )

    finish_btn.click(
        on_finish_task,
        [session_state],
        [finish_out, progress_bar, step3_col, step4_col, survey_session_info],
    )

    # Step 4 → 5: submit survey, show done screen
    def on_submit_survey(completed, time_val, answer_text, confidence,
                         toast_r, toast_c, toast_t,
                         tlx_mental, tlx_effort,
                         manipulation, verified, comments, state):
        pid  = state.get("participant_id", "?")
        cond = state.get("condition", "?")
        tid  = state.get("task_id", "?")
        msg = submit_survey(
            pid, cond, tid, completed, time_val,
            answer_text, confidence,
            toast_r, toast_c, toast_t,
            tlx_mental, tlx_effort,
            manipulation, verified, comments, state,
        )
        return (
            msg,
            gr.update(value=make_progress(5)),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    survey_btn.click(
        on_submit_survey,
        [s_completed, s_time, s_answer_text, s_confidence,
         s_toast_reliable, s_toast_confident, s_toast_trustworthy,
         s_tlx_mental, s_tlx_effort,
         s_manipulation, s_verified, s_comments, session_state],
        [survey_out, progress_bar, step4_col, step5_col],
    )

    # Condition A: search
    search_btn.click(search_condition_a, [search_box, session_state], [search_results, session_state])
    search_box.submit(search_condition_a, [search_box, session_state], [search_results, session_state])

    # Condition B: chat
    chat_btn.click(
        chat_condition_b,
        [chat_box, chat_history, session_state],
        [chat_box, chatbot, session_state, chat_history],
    )
    chat_box.submit(
        chat_condition_b,
        [chat_box, chat_history, session_state],
        [chat_box, chatbot, session_state, chat_history],
    )
    clear_btn.click(lambda: ([], []), None, [chatbot, chat_history])

    # Researcher view
    def unlock(password):
        if password == RESEARCHER_PASSWORD:
            return (
                "✅ Access granted.",
                gr.update(visible=True, value=load_log()),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        return (
            "❌ Wrong password.",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    unlock_btn.click(unlock, [researcher_password], [access_msg, log_display, button_row, download_file])
    refresh_btn.click(load_log, None, log_display)
    download_btn.click(download_log, None, download_file)

demo.launch()
