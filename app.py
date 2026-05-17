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

# ── Experiment Tasks (hints removed — they asymmetrically help Condition A) ───
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

# FIX: Use new messages format ({"role": ..., "content": ...}) instead of tuples.
# gr.Chatbot(type="messages") requires this format; tuples are deprecated in recent Gradio.
def chat_condition_b(message, history, state):
    if not message.strip():
        return "", history, state, history

    state["query_count"] = state.get("query_count", 0) + 1
    state["queries"] = state.get("queries", []) + [message]
    t_start = time.time()

    # Build contents from new-format history (list of {"role": ..., "content": ...})
    contents = []
    for turn in (history or []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            contents.append(types.Content(role="user",  parts=[types.Part(text=str(content))]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=str(content))]))

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

        # FIX: Log full response text (not just length) for post-hoc hallucination analysis
        log_event(state.get("participant_id", "unknown"), "B", "chat", {
            "query":            message,
            "query_length":     len(message),
            "query_number":     state.get("query_count", 0),
            "response_text":    answer,          # full text for H3 hallucination analysis
            "response_length":  len(answer),
            "sources":          sources,
            "elapsed_s":        elapsed,
        })

    except Exception as e:
        answer = f"Error: {e}"

    # Build new history in messages format
    new_history = list(history or []) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", new_history, state, new_history

# ── Pre-task survey submission ────────────────────────────────────────────────
def submit_pre_survey(pid, age, education, language, museum_familiarity,
                      ai_usage, aias1, aias2, aias3, aias4, search_comfort, state):
    data = {
        "participant_id":    pid,
        "age":               age,
        "education":         education,
        "native_language":   language,
        "museum_familiarity": museum_familiarity,
        "ai_usage_freq":     ai_usage,
        "aias4_item1":       aias1,
        "aias4_item2":       aias2,
        "aias4_item3":       aias3,
        "aias4_item4":       aias4,
        "search_comfort":    search_comfort,
    }
    log_event(pid, state.get("condition", "?"), "pre_survey", data)
    return f"✅ Pre-task survey saved for **{pid}**. You may now proceed to the Setup tab."

# ── Post-task survey submission ───────────────────────────────────────────────
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
        # H2: accuracy
        "participant_answer":     q_answer_text,
        "answer_confidence":      q_confidence,
        # H3: trust calibration
        "toast_reliable":         q_toast_reliable,
        "toast_confident":        q_toast_confident,
        "toast_trustworthy":      q_toast_trustworthy,
        # H4: effort/cognitive load
        "tlx_mental_demand":      q_tlx_mental,
        "tlx_effort":             q_tlx_effort,
        # manipulation check
        "manipulation_check":     q_manipulation_check,
        # qualitative
        "verified_sources":       q_verified,
        "comments":               q_comments,
        # behavioural
        "query_count":            state.get("query_count", 0),
        "queries":                state.get("queries", []),
    }
    log_event(pid, condition, "survey", survey_data)
    return (
        f"Survey submitted! Thank you, {pid}.\n\n"
        f"Condition: {condition} · Task: {task_id} · Queries made: {state.get('query_count', 0)}\n\n"
        f"Your responses have been saved to {LOG_FILE}."
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
    return f"✅ Session ended for **{state.get('participant_id', '?')}**. Total time: {elapsed}s · Total queries: {state.get('query_count', 0)}"

# ── Download / load log ───────────────────────────────────────────────────────
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

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Museum Collection Experiment") as demo:

    session_state = gr.State({})
    chat_history  = gr.State([])

    gr.Markdown("# Museum Collection Study\nUser Study — Please complete the Pre-Task Survey first, then proceed to Setup.")

    # ── Tab 0: Pre-Task Survey ────────────────────────────────────────────────
    with gr.Tab("Pre-Task Survey"):
        gr.Markdown(
            "### Before you begin\n"
            "Please answer these short questions. They help us control for individual differences in the analysis.\n"
            "All responses are anonymous."
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

        gr.Markdown("#### Domain & Tool Familiarity")
        pre_museum   = gr.Slider(1, 5, step=1, value=3,
                                 label="Museum / art history familiarity (1 = none, 5 = expert)")
        pre_ai_usage = gr.Dropdown(
            label="How often do you use ChatGPT-style AI tools?",
            choices=["Never", "Rarely (a few times a year)", "Sometimes (monthly)",
                     "Often (weekly)", "Daily"],
        )
        pre_search   = gr.Slider(1, 5, step=1, value=3,
                                 label="Comfort searching databases / library catalogues (1 = not at all, 5 = very comfortable)")

        gr.Markdown(
            "#### Attitude towards AI (AIAS-4)\n"
            "Rate each statement from 1 (strongly disagree) to 5 (strongly agree). "
            "*(Grassini, 2023)*"
        )
        with gr.Row():
            pre_aias1 = gr.Slider(1, 5, step=1, value=3,
                                  label="AI systems are capable of performing tasks as well as humans")
            pre_aias2 = gr.Slider(1, 5, step=1, value=3,
                                  label="I feel comfortable relying on AI for information")
        with gr.Row():
            pre_aias3 = gr.Slider(1, 5, step=1, value=3,
                                  label="AI tools are a useful addition to everyday work")
            pre_aias4 = gr.Slider(1, 5, step=1, value=3,
                                  label="I trust AI-generated results to be mostly accurate")

        pre_submit_btn = gr.Button("Save Pre-Task Survey", variant="primary")
        pre_submit_out = gr.Markdown("")

        pre_submit_btn.click(
            submit_pre_survey,
            [pre_pid, pre_age, pre_education, pre_language,
             pre_museum, pre_ai_usage,
             pre_aias1, pre_aias2, pre_aias3, pre_aias4,
             pre_search, session_state],
            [pre_submit_out],
        )

    # ── Tab 1: Setup ──────────────────────────────────────────────────────────
    with gr.Tab("Setup"):
        gr.Markdown("### Participant Setup\nEnter your details to begin.")

        with gr.Row():
            pid_box  = gr.Textbox(label="Participant ID (e.g. P01)", placeholder="P01")
            cond_box = gr.Dropdown(
                label="Condition (assigned by researcher)",
                choices=["A — Keyword Search", "B — AI Chat"],
                value="A — Keyword Search"
            )

        task_dropdown = gr.Dropdown(
            label="Select your task",
            choices=[f"Task {t['id']}: {t['title']}" for t in TASKS],
            value="Task 1: Find a ritual object",
        )

        setup_btn = gr.Button("Start Session", variant="primary")
        setup_out = gr.Markdown("")

        finish_btn = gr.Button("Finish Task", variant="secondary")
        finish_out = gr.Markdown("")

        def start_session(pid, cond, task_label, state):
            task_id = int(task_label.split(":")[0].replace("Task ", "").strip())
            task = next(t for t in TASKS if t["id"] == task_id)
            state["participant_id"] = pid
            state["condition"]      = cond
            state["task_id"]        = task_id
            state["session_start"]  = time.time()
            state["query_count"]    = 0
            state["queries"]        = []
            log_event(pid, cond, "session_start", {"task_id": task_id})
            tab_name = "Keyword Search" if "A" in cond else "AI Chat"
            return (
                f"Session started for **{pid}** · Condition {cond} · Task {task_id}\n\n"
                f"**Your task:** {task['description']}\n\n"
                f"Now go to the **{tab_name}** tab."
            ), state

        setup_btn.click(start_session, [pid_box, cond_box, task_dropdown, session_state],
                        [setup_out, session_state])
        finish_btn.click(end_session, [session_state], [finish_out])

    # ── Tab 2: Keyword Search ─────────────────────────────────────────────────
    with gr.Tab("Keyword Search (Condition A)"):
        gr.Markdown("### Condition A — Simple Keyword Search\nType keywords to search the museum collection. Results are direct matches only — no AI interpretation.")

        with gr.Row():
            search_box = gr.Textbox(
                placeholder="e.g. ritual object, 1700, lacquer...",
                show_label=False, scale=8
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        search_results = gr.HTML("<p style='color:#888;padding:20px;'>Results will appear here.</p>")

        search_btn.click(search_condition_a, [search_box, session_state], [search_results, session_state])
        search_box.submit(search_condition_a, [search_box, session_state], [search_results, session_state])

    # ── Tab 3: AI Chat ────────────────────────────────────────────────────────
    with gr.Tab("AI Chat (Condition B)"):
        gr.Markdown("### Condition B — AI-Powered Research Assistant\nAsk questions in natural language. AI answers may contain errors — always verify with source links.")

        # FIX: type="messages" → expects {"role": ..., "content": ...} dicts, not tuples
        chatbot = gr.Chatbot(label="Chat", height=480, type="messages")

        with gr.Row():
            chat_box = gr.Textbox(
                placeholder="Ask about the collection...",
                show_label=False, scale=9
            )
            chat_btn = gr.Button("Send", variant="primary", scale=1)

        clear_btn = gr.Button("Clear chat", size="sm")

        chat_btn.click(chat_condition_b, [chat_box, chat_history, session_state],
                       [chat_box, chatbot, session_state, chat_history])
        chat_box.submit(chat_condition_b, [chat_box, chat_history, session_state],
                        [chat_box, chatbot, session_state, chat_history])
        clear_btn.click(lambda: ([], []), None, [chatbot, chat_history])

    # ── Tab 4: Tasks ──────────────────────────────────────────────────────────
    with gr.Tab("Tasks"):
        gr.Markdown("### Experiment Task Cards\nRead your assigned task carefully before starting.")
        for t in TASKS:
            with gr.Accordion(f"Task {t['id']}: {t['title']}", open=(t['id'] == 1)):
                gr.Markdown(f"**Description:** {t['description']}")

    # ── Tab 5: Survey ─────────────────────────────────────────────────────────
    with gr.Tab("Survey"):
        gr.Markdown(
            "### Post-Task Survey\n"
            "Complete this after finishing your task.\n"
            "Sliders: 1 = strongly disagree / not at all · 7 = strongly agree / extremely."
        )

        with gr.Row():
            s_pid  = gr.Textbox(label="Your Participant ID")
            s_cond = gr.Dropdown(label="Your Condition", choices=["A — Keyword Search", "B — AI Chat"])
            s_task = gr.Dropdown(label="Task you completed",
                                 choices=[f"Task {t['id']}: {t['title']}" for t in TASKS])

        with gr.Row():
            s_completed = gr.Radio(label="Did you complete the task?", choices=["Yes", "Partially", "No"])
            s_time      = gr.Number(label="Approx. time taken (minutes)", value=5)

        gr.Markdown("#### Your Answer (H2 — Accuracy)")
        s_answer_text = gr.Textbox(
            label="What is your final answer to the task? (write it out fully)",
            lines=3,
            placeholder="e.g. The ritual object is called X, made in year Y, from culture Z."
        )
        s_confidence = gr.Slider(1, 7, step=1, value=4,
                                 label="How confident are you that your answer is correct? (1 = not at all, 7 = completely sure)")

        gr.Markdown("#### TOAST Trust Scale (H3)")
        with gr.Row():
            s_toast_reliable    = gr.Slider(1, 7, step=1, value=4, label="The system performed reliably")
            s_toast_confident   = gr.Slider(1, 7, step=1, value=4, label="I felt confident using this system")
            s_toast_trustworthy = gr.Slider(1, 7, step=1, value=4, label="I found the system trustworthy")

        gr.Markdown("#### Cognitive Load — NASA-TLX (H4)")
        with gr.Row():
            s_tlx_mental = gr.Slider(1, 7, step=1, value=4,
                                     label="How mentally demanding was the task? (1 = not at all, 7 = extremely)")
            s_tlx_effort = gr.Slider(1, 7, step=1, value=4,
                                     label="How hard did you have to work to accomplish your performance? (1 = very little, 7 = very hard)")

        gr.Markdown("#### Manipulation Check")
        s_manipulation = gr.Radio(
            label="What kind of tool did you just use?",
            choices=["Keyword search", "AI assistant", "Both", "Not sure"]
        )

        gr.Markdown("#### Critical Evaluation")
        s_verified = gr.Radio(
            label="Did you verify any answers with the source link?",
            choices=["Yes, always", "Sometimes", "No"]
        )

        s_comments = gr.Textbox(label="Any comments or feedback? (optional)", lines=3,
                                placeholder="What worked well? What was frustrating?")

        survey_btn = gr.Button("Submit Survey", variant="primary")
        survey_out = gr.Markdown("")

        survey_btn.click(
            submit_survey,
            [s_pid, s_cond, s_task, s_completed, s_time,
             s_answer_text, s_confidence,
             s_toast_reliable, s_toast_confident, s_toast_trustworthy,
             s_tlx_mental, s_tlx_effort,
             s_manipulation,
             s_verified, s_comments, session_state],
            [survey_out],
        )

    # ── Tab 6: Researcher View (password protected) ───────────────────────────
    with gr.Tab("Researcher View"):
        gr.Markdown("### Researcher Access Only\nThis area is for the researcher. Participants do not need this tab.")

        researcher_password = gr.Textbox(
            label="Enter researcher password",
            type="password",
            placeholder="Enter password to unlock"
        )
        unlock_btn = gr.Button("Unlock", variant="primary")
        access_msg = gr.Markdown("")

        log_display   = gr.Code(label="experiment_log.jsonl", language="json", lines=30, visible=False)
        with gr.Row(visible=False) as button_row:
            refresh_btn  = gr.Button("Refresh Log", variant="secondary")
            download_btn = gr.Button("⬇️ Download Log File", variant="primary")
        download_file = gr.File(label="Your download will appear here", visible=False)

        def unlock(password):
            if password == RESEARCHER_PASSWORD:
                return (
                    "✅ Access granted. Welcome, researcher.",
                    gr.update(visible=True, value=load_log()),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )
            else:
                return (
                    "❌ Wrong password. Try again.",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

        unlock_btn.click(
            unlock,
            [researcher_password],
            [access_msg, log_display, button_row, download_file]
        )

        refresh_btn.click(load_log, None, log_display)
        download_btn.click(download_log, None, download_file)

demo.launch()