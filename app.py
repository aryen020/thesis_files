import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "google-genai"], check=True)

import os
import json
import time
import random
import datetime
import pandas as pd
import gradio as gr
import google.genai as genai
from google.genai import types

# ── Config ───────────────────────────────────────────────────────────────
API_KEY             = os.environ["GEMINI_API_KEY"]
STORE_NAME          = os.environ.get("STORE_NAME", "fileSearchStores/userstudystore-3gytybx82f4t")
MODEL               = "gemini-2.5-flash"
LOG_FILE            = "experiment_log.jsonl"
CSV_PATH            = "small_dataset.csv"

client = genai.Client(api_key=API_KEY)

file_search_tool = types.Tool(
    file_search=types.FileSearch(
        file_search_store_names=[STORE_NAME]
    )
)

# ── Tasks ────────────────────────────────────────────────────────────────
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

# ── Logging ──────────────────────────────────────────────────────────────
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

# ── Condition Assignment ─────────────────────────────────────────────────
def assign_condition(pid):
    random.seed(pid)
    return random.choice(["A", "B"])

# ── Keyword Search ───────────────────────────────────────────────────────
def keyword_search(query):
    results = []

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8", encoding_errors="replace")

        keywords = query.lower().split()

        mask = df.apply(
            lambda row: any(
                kw in " ".join(row.astype(str).str.lower())
                for kw in keywords
            ),
            axis=1
        )

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
        results = [{
            "title": f"Error: {e}",
            "type": "",
            "creator": "",
            "date": "",
            "url": "",
            "image": "",
        }]

    return results

# ── Condition A Search ───────────────────────────────────────────────────
def search_condition_a(query, state):

    if not query.strip():
        return "<p>Enter a search term.</p>", state

    t_start = time.time()

    results = keyword_search(query)

    elapsed = round(time.time() - t_start, 2)

    state["query_count"] += 1
    state["queries"].append(query)

    log_event(
        state["participant_id"],
        "A",
        "search",
        {
            "query": query,
            "elapsed_s": elapsed,
            "results": len(results)
        }
    )

    cards = ""

    for r in results:

        img_html = ""

        if r["image"] and r["image"] != "nan":
            img_html = f"""
            <img src="{r['image']}"
                 style="width:90px;height:90px;object-fit:cover;border-radius:8px;">
            """

        cards += f"""
        <div style="
            border:1px solid #ddd;
            border-radius:12px;
            padding:14px;
            margin-bottom:12px;
            display:flex;
            gap:14px;
            background:white;
        ">
            {img_html}

            <div>
                <div style="font-weight:700;font-size:17px;">
                    {r['title']}
                </div>

                <div style="font-size:13px;color:#666;margin-top:4px;">
                    Type: {r['type']}<br>
                    Creator: {r['creator']}<br>
                    Date: {r['date']}
                </div>

                <div style="margin-top:8px;">
                    <a href="{r['url']}" target="_blank">
                        View Record
                    </a>
                </div>
            </div>
        </div>
        """

    html = f"""
    <div style="margin-bottom:10px;color:#666;">
        {len(results)} results · {elapsed}s
    </div>

    {cards}
    """

    return html, state

# ── Gemini Chat ──────────────────────────────────────────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum research assistant. "
    "Answer using the indexed museum collection. "
    "If unsure, say so explicitly."
)

def _to_gemini_contents(history):

    contents = []

    for turn in history:

        if turn["role"] == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=turn["content"])]
                )
            )

        elif turn["role"] == "assistant":
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=turn["content"])]
                )
            )

    return contents

# ── Condition B Chat ─────────────────────────────────────────────────────
def chat_condition_b(message, history, state):

    if not message.strip():
        return "", history, state, history

    state["query_count"] += 1
    state["queries"].append(message)

    contents = _to_gemini_contents(history)

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    )

    t_start = time.time()

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

    elapsed = round(time.time() - t_start, 2)

    log_event(
        state["participant_id"],
        "B",
        "chat",
        {
            "query": message,
            "response": answer,
            "elapsed_s": elapsed,
        }
    )

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    return "", new_history, state, new_history

# ── Start Study ──────────────────────────────────────────────────────────
def start_study(pid, age, ai_usage, state):

    condition = assign_condition(pid)

    task = random.choice(TASKS)

    state["participant_id"] = pid
    state["condition"] = condition
    state["task"] = task
    state["session_start"] = time.time()
    state["query_count"] = 0
    state["queries"] = []

    log_event(
        pid,
        condition,
        "session_start",
        {
            "task_id": task["id"],
            "age": age,
            "ai_usage": ai_usage,
        }
    )

    task_html = f"""
    <div style="
        padding:20px;
        border-radius:14px;
        background:#f8f8f8;
        border:1px solid #ddd;
    ">

    <div style="font-size:14px;color:#666;">
        Assigned Condition: <b>{condition}</b>
    </div>

    <h2>
        Task {task['id']} — {task['title']}
    </h2>

    <p style="font-size:18px;">
        {task['description']}
    </p>

    </div>
    """

    show_search = condition == "A"
    show_chat = condition == "B"

    return (
        state,
        task_html,

        gr.update(visible=False),
        gr.update(visible=True),

        gr.update(visible=show_search),
        gr.update(visible=show_chat),
    )

# ── Finish Study ─────────────────────────────────────────────────────────
def finish_study(confidence, comments, state):

    elapsed = round(
        time.time() - state["session_start"],
        1
    )

    log_event(
        state["participant_id"],
        state["condition"],
        "final_survey",
        {
            "confidence": confidence,
            "comments": comments,
            "queries": state["queries"],
            "query_count": state["query_count"],
            "time_s": elapsed,
        }
    )

    return f"""
    # Thank You

    Your responses have been recorded.

    Total queries: {state['query_count']}

    Total time: {elapsed} seconds.
    """

# ── UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Museum Collection Study",
    theme=gr.themes.Soft()
) as demo:

    state = gr.State({})
    history = gr.State([])

    # ── Welcome Screen ────────────────────────────────────────────────
    with gr.Column(visible=True) as welcome_screen:

        gr.Markdown("""
        # Museum Collection Study

        Welcome.

        This study takes approximately 10 minutes.

        You will complete:
        - a short pre-survey
        - one research task
        - a short final survey
        """)

        pid_input = gr.Textbox(
            label="Participant ID",
            placeholder="e.g. P01"
        )

        age_input = gr.Number(
            label="Age",
            value=25
        )

        ai_usage_input = gr.Dropdown(
            label="How often do you use AI tools?",
            choices=[
                "Never",
                "Rarely",
                "Sometimes",
                "Often",
                "Daily"
            ]
        )

        start_btn = gr.Button(
            "Start Study",
            variant="primary",
            size="lg"
        )

    # ── Main Experiment Screen ───────────────────────────────────────
    with gr.Column(visible=False) as experiment_screen:

        task_display = gr.HTML()

        gr.Markdown("---")

        # CONDITION A
        with gr.Column(visible=False) as keyword_ui:

            gr.Markdown("## Search Interface")

            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search the collection..."
                )

                search_btn = gr.Button(
                    "Search",
                    variant="primary"
                )

            search_results = gr.HTML()

        # CONDITION B
        with gr.Column(visible=False) as chat_ui:

            gr.Markdown("## AI Research Assistant")

            chatbot = gr.Chatbot(
                height=450,
                type="messages"
            )

            with gr.Row():

                chat_box = gr.Textbox(
                    placeholder="Ask a question..."
                )

                chat_btn = gr.Button(
                    "Send",
                    variant="primary"
                )

        gr.Markdown("---")

        continue_btn = gr.Button(
            "Continue to Final Survey",
            variant="secondary"
        )

    # ── Final Survey ────────────────────────────────────────────────
    with gr.Column(visible=False) as survey_screen:

        gr.Markdown("# Final Survey")

        confidence_slider = gr.Slider(
            1,
            7,
            value=4,
            step=1,
            label="How confident are you in your answers?"
        )

        comments_box = gr.Textbox(
            lines=4,
            label="Comments or feedback"
        )

        submit_btn = gr.Button(
            "Submit Study",
            variant="primary",
            size="lg"
        )

    # ── Thank You Screen ────────────────────────────────────────────
    with gr.Column(visible=False) as thankyou_screen:

        thankyou_text = gr.Markdown()

    # ── Navigation ─────────────────────────────────────────────────
    start_btn.click(
        start_study,
        inputs=[
            pid_input,
            age_input,
            ai_usage_input,
            state
        ],
        outputs=[
            state,
            task_display,

            welcome_screen,
            experiment_screen,

            keyword_ui,
            chat_ui,
        ]
    )

    continue_btn.click(
        lambda: (
            gr.update(visible=False),
            gr.update(visible=True)
        ),
        outputs=[
            experiment_screen,
            survey_screen
        ]
    )

    submit_btn.click(
        finish_study,
        inputs=[
            confidence_slider,
            comments_box,
            state
        ],
        outputs=[
            thankyou_text
        ]
    ).then(
        lambda: (
            gr.update(visible=False),
            gr.update(visible=True)
        ),
        outputs=[
            survey_screen,
            thankyou_screen
        ]
    )

    # ── Search Events ──────────────────────────────────────────────
    search_btn.click(
        search_condition_a,
        inputs=[search_box, state],
        outputs=[search_results, state]
    )

    search_box.submit(
        search_condition_a,
        inputs=[search_box, state],
        outputs=[search_results, state]
    )

    # ── Chat Events ────────────────────────────────────────────────
    chat_btn.click(
        chat_condition_b,
        inputs=[chat_box, history, state],
        outputs=[chat_box, chatbot, state, history]
    )

    chat_box.submit(
        chat_condition_b,
        inputs=[chat_box, history, state],
        outputs=[chat_box, chatbot, state, history]
    )

# ── Launch ───────────────────────────────────────────────────────────────
demo.launch()