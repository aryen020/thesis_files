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

# ── Config ─────────────────────────────────────────────────────────────
API_KEY             = os.environ["GEMINI_API_KEY"]
STORE_NAME          = os.environ.get(
    "STORE_NAME",
    "fileSearchStores/userstudystore-3gytybx82f4t"
)

RESEARCHER_PASSWORD = os.environ.get(
    "RESEARCHER_PASSWORD",
    "mypassword123"
)

MODEL       = "gemini-2.5-flash"
LOG_FILE    = "experiment_log.jsonl"
CSV_PATH    = "small_dataset.csv"

client = genai.Client(api_key=API_KEY)

file_search_tool = types.Tool(
    file_search=types.FileSearch(
        file_search_store_names=[STORE_NAME]
    )
)

# ── Tasks ──────────────────────────────────────────────────────────────
TASKS = [
    {
        "id": 1,
        "title": "Find a ritual object",
        "description": (
            "Find any ritual object in the collection. "
            "What is it called, when was it made, "
            "and what culture does it come from?"
        ),
    },
    {
        "id": 2,
        "title": "Identify the oldest item",
        "description": (
            "What is the oldest item in the dataset? "
            "Provide its name, ID, and creation date."
        ),
    },
    {
        "id": 3,
        "title": "Find lacquerware",
        "description": (
            "Find two examples of lacquerware "
            "in the collection. Who made them and when?"
        ),
    },
    {
        "id": 4,
        "title": "Artworks from 1700",
        "description": (
            "How many items in the collection were "
            "created around 1700? List at least three."
        ),
    },
    {
        "id": 5,
        "title": "Anonymous creators",
        "description": (
            "Find three items created by anonymous makers. "
            "What types of objects are they?"
        ),
    },
]

# ── Logging ────────────────────────────────────────────────────────────
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

# ── Assign Condition ───────────────────────────────────────────────────
def assign_condition(pid):

    random.seed(pid)

    return random.choice(["A", "B"])

# ── Keyword Search ─────────────────────────────────────────────────────
def keyword_search(query):

    results = []

    try:

        df = pd.read_csv(
            CSV_PATH,
            encoding="utf-8",
            encoding_errors="replace"
        )

        keywords = query.lower().split()

        mask = df.apply(
            lambda row: any(
                kw in " ".join(
                    row.astype(str).str.lower()
                )
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

# ── Condition A ────────────────────────────────────────────────────────
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
            "result_count": len(results),
        }
    )

    cards = ""

    for r in results:

        img_html = ""

        if r["image"] and r["image"] != "nan":

            img_html = f"""
            <img src="{r['image']}"
                 style="
                    width:90px;
                    height:90px;
                    object-fit:cover;
                    border-radius:8px;
                 ">
            """

        link = ""

        if r["url"] and r["url"] != "nan":

            link = f"""
            <a href="{r['url']}"
               target="_blank"
               style="color:#c77d3a;">
                View Record
            </a>
            """

        cards += f"""
        <div style="
            display:flex;
            gap:14px;
            background:white;
            border:1px solid #ddd;
            border-radius:12px;
            padding:14px;
            margin-bottom:12px;
        ">

            {img_html}

            <div>

                <div style="
                    font-weight:700;
                    font-size:17px;
                ">
                    {r['title']}
                </div>

                <div style="
                    color:#666;
                    font-size:13px;
                    margin-top:5px;
                ">
                    Type: {r['type']}<br>
                    Creator: {r['creator']}<br>
                    Date: {r['date']}
                </div>

                <div style="margin-top:8px;">
                    {link}
                </div>

            </div>

        </div>
        """

    html = f"""
    <div style="
        color:#777;
        margin-bottom:12px;
    ">
        {len(results)} result(s) · {elapsed}s
    </div>

    {cards}
    """

    return html, state

# ── Gemini History Conversion ──────────────────────────────────────────
def _to_gemini_contents(history):

    contents = []

    for user_msg, bot_msg in history:

        if user_msg:

            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=str(user_msg))]
                )
            )

        if bot_msg:

            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=str(bot_msg))]
                )
            )

    return contents

# ── Condition B ────────────────────────────────────────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum research assistant. "
    "Answer questions using the indexed collection documents. "
    "Be clear and informative. "
    "If unsure, explicitly say so."
)

def chat_condition_b(message, history, state):

    if not message.strip():
        return "", history, state, history

    t_start = time.time()

    state["query_count"] += 1
    state["queries"].append(message)

    contents = _to_gemini_contents(history)

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=message)]
        )
    )

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
        (message, answer)
    ]

    return "", new_history, state, new_history

# ── Start Session ──────────────────────────────────────────────────────
def start_session(pid, age, ai_usage, state):

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
        background:#f9f9f9;
        border:1px solid #ddd;
        border-radius:14px;
        padding:20px;
        margin-bottom:18px;
    ">

        <div style="
            color:#666;
            font-size:13px;
            margin-bottom:8px;
        ">
            Assigned Condition: <b>{condition}</b>
        </div>

        <div style="
            font-size:26px;
            font-weight:700;
            margin-bottom:10px;
        ">
            Task {task['id']} — {task['title']}
        </div>

        <div style="
            font-size:18px;
            line-height:1.6;
        ">
            {task['description']}
        </div>

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

# ── Finish Study ───────────────────────────────────────────────────────
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

Your study session has been recorded successfully.

### Session Summary

- Condition: {state['condition']}
- Queries made: {state['query_count']}
- Time spent: {elapsed} seconds

You may now close this page.
"""

# ── UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Museum Collection Study",
    theme=gr.themes.Soft(),
    fill_height=True,
) as demo:

    session_state = gr.State({})
    chat_history = gr.State([])

    # ── Welcome Screen ─────────────────────────────────────────────
    with gr.Column(visible=True) as welcome_screen:

        gr.Markdown("""
        # Museum Collection Study

        Welcome.

        This study takes approximately 10 minutes.

        You will:
        - complete a short questionnaire
        - complete one research task
        - answer a short final survey
        """)

        with gr.Row():

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
            ],
        )

        start_btn = gr.Button(
            "Start Study",
            variant="primary",
            size="lg"
        )

    # ── Experiment Screen ─────────────────────────────────────────
    with gr.Column(visible=False) as experiment_screen:

        progress = gr.Markdown("""
        ### Step 2 of 3 — Complete Your Research Task
        """)

        task_display = gr.HTML()

        gr.Info(
            "Use the interface below to complete your assigned task. "
            "You may search as many times as needed."
        )

        # ── Condition A UI ─────────────────────────────────────
        with gr.Column(visible=False) as keyword_ui:

            gr.Markdown("## Keyword Search")

            with gr.Row(equal_height=True):

                search_box = gr.Textbox(
                    placeholder="Search the collection...",
                    scale=8
                )

                search_btn = gr.Button(
                    "Search",
                    variant="primary",
                    size="lg",
                    scale=1
                )

            search_results = gr.HTML(
                "<p style='color:#888'>Results will appear here.</p>"
            )

        # ── Condition B UI ─────────────────────────────────────
        with gr.Column(visible=False) as chat_ui:

            gr.Markdown("## AI Research Assistant")

            chatbot = gr.Chatbot(
                height=550,
                bubble_full_width=False
            )

            with gr.Row(equal_height=True):

                chat_box = gr.Textbox(
                    placeholder="Ask about the collection...",
                    scale=8
                )

                chat_btn = gr.Button(
                    "Send",
                    variant="primary",
                    size="lg",
                    scale=1
                )

        gr.Markdown("---")

        continue_btn = gr.Button(
            "Continue to Final Survey →",
            variant="secondary",
            size="lg"
        )

    # ── Final Survey ────────────────────────────────────────────
    with gr.Column(visible=False) as survey_screen:

        gr.Markdown("""
        ### Step 3 of 3 — Final Survey
        """)

        confidence_slider = gr.Slider(
            1,
            7,
            step=1,
            value=4,
            label="How confident are you in your final answers?"
        )

        comments_box = gr.Textbox(
            lines=4,
            label="Comments or feedback (optional)"
        )

        submit_btn = gr.Button(
            "Submit Study",
            variant="primary",
            size="lg"
        )

    # ── Thank You Screen ────────────────────────────────────────
    with gr.Column(visible=False) as thankyou_screen:

        thankyou_text = gr.Markdown()

    # ── Navigation ──────────────────────────────────────────────
    start_btn.click(
        start_session,
        inputs=[
            pid_input,
            age_input,
            ai_usage_input,
            session_state
        ],
        outputs=[
            session_state,
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
            session_state
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

    # ── Search Events ───────────────────────────────────────────
    search_btn.click(
        search_condition_a,
        inputs=[
            search_box,
            session_state
        ],
        outputs=[
            search_results,
            session_state
        ]
    )

    search_box.submit(
        search_condition_a,
        inputs=[
            search_box,
            session_state
        ],
        outputs=[
            search_results,
            session_state
        ]
    )

    # ── Chat Events ─────────────────────────────────────────────
    chat_btn.click(
        chat_condition_b,
        inputs=[
            chat_box,
            chat_history,
            session_state
        ],
        outputs=[
            chat_box,
            chatbot,
            session_state,
            chat_history
        ]
    )

    chat_box.submit(
        chat_condition_b,
        inputs=[
            chat_box,
            chat_history,
            session_state
        ],
        outputs=[
            chat_box,
            chatbot,
            session_state,
            chat_history
        ]
    )

# ── Launch ─────────────────────────────────────────────────────────────
demo.launch()