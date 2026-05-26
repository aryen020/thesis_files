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
    {"id": 1, "title": "Ritueel object",
     "description": "Zoek een ritueel object in de collectie. Hoe heet het, wanneer is het gemaakt, en uit welke cultuur komt het?"},
    {"id": 2, "title": "Oudste object",
     "description": "Wat is het oudste object in de collectie? Geef de naam, het ID en de vervaardigingsdatum."},
    {"id": 3, "title": "Lakwerk",
     "description": "Zoek twee voorbeelden van lakwerk in de collectie. Door wie zijn ze gemaakt en wanneer?"},
    {"id": 4, "title": "Kunstwerken uit circa 1700",
     "description": "Hoeveel objecten in de collectie zijn rond 1700 gemaakt? Noem er minimaal drie."},
    {"id": 5, "title": "Anonieme makers",
     "description": "Zoek drie objecten van anonieme makers. Wat voor soort objecten zijn het?"},
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

# ── Conditie A: trefwoordzoeken ───────────────────────────────────────────────
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
        results = [{"title": f"Fout: {e}", "id": "", "type": "", "creator": "", "date": "", "url": "", "image": ""}]
    return results

def search_condition_a(query, state):
    if not query.strip():
        return "<p style='color:#888'>Voer een zoekterm in.</p>", state
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
        return "<p style='color:#e07b39;'>Geen resultaten gevonden. Probeer een andere zoekterm.</p>", state
    cards = ""
    for r in results:
        img = (f"<img src='{r['image']}' style='width:80px;height:80px;object-fit:cover;"
               f"border-radius:6px;flex-shrink:0;' onerror=\"this.style.display='none'\">"
               if r["image"] and r["image"] != "nan" else "")
        link = (f"<a href='{r['url']}' target='_blank' style='color:#c77d3a;font-size:11px;'>Bekijk record</a>"
                if r["url"] and r["url"] != "nan" else "")
        cards += f"""
<div style='display:flex;gap:12px;align-items:flex-start;background:#f9f9f9;
            border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:10px;'>
  {img}
  <div>
    <div style='font-weight:600;font-size:15px;'>{r['title']}</div>
    <div style='color:#666;font-size:12px;margin-top:4px;'>
      Type: {r['type']} | Maker: {r['creator']} | Datum: {r['date']}
    </div>
    <div style='margin-top:6px;'>{link}</div>
  </div>
</div>"""
    header = (f"<div style='font-size:12px;color:#888;margin-bottom:10px;'>"
              f"{len(results)} resultaat/resultaten voor \"{query}\" · {elapsed}s · trefwoordzoekopdracht</div>")
    return header + cards, state

# ── Conditie B: RAG-chat ──────────────────────────────────────────────────────
SYSTEM_PROMPT_B = (
    "You are a helpful museum research assistant. Answer questions using the indexed collection documents. "
    "Be clear and informative. If you are unsure or the document does not contain the answer, say so explicitly. "
    "Always mention which source document you used."
)

def _to_gemini_contents(history):
    contents = []
    for turn in (history or []):
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_msg, bot_msg = turn
            if user_msg:
                contents.append(types.Content(role="user",  parts=[types.Part(text=str(user_msg))]))
            if bot_msg:
                contents.append(types.Content(role="model", parts=[types.Part(text=str(bot_msg))]))
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
        answer  = response.text or "Geen antwoord gegenereerd."
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
            answer += "\n\n⚠️ Let op: De AI gaf aan beperkte zekerheid te hebben. Verifieer met de originele bron."
        if sources:
            answer += "\n\nBronnen: " + " · ".join(s.replace('.txt','').replace('.pdf','') for s in sources)
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
        answer = f"Fout: {e}"
    new_history = list(history or []) + [[message, answer]]
    return "", new_history, state, new_history

# ── Voorafgaande enquête ──────────────────────────────────────────────────────
def submit_pre_survey(pid, age, edu, lang, museum, ai_use, search_c, a1, a2, a3, a4, state):
    log_event(pid, "?", "pre_survey", {
        "participant_id": pid, "age": age, "education": edu,
        "native_language": lang, "museum_familiarity": museum, "ai_usage_freq": ai_use,
        "search_comfort": search_c,
        "aias4_item1": a1, "aias4_item2": a2, "aias4_item3": a3, "aias4_item4": a4,
    })

# ── Eindsurvey ────────────────────────────────────────────────────────────────
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
        return "Nog geen log-invoer."

def download_log():
    return LOG_FILE if os.path.exists(LOG_FILE) else None

# ── UI-helpers ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.stepper{display:flex;gap:0;margin-bottom:28px;font-size:13px;font-weight:500;}
.step{flex:1;text-align:center;padding:10px 4px;background:#e2e8f0;color:#64748b;border-right:2px solid white;}
.step:first-child{border-radius:8px 0 0 8px;}
.step:last-child{border-radius:0 8px 8px 0;border-right:none;}
.step.active{background:#3b82f6;color:white;font-weight:700;}
.step.done{background:#bbf7d0;color:#166534;}
.task-card{background:#eff6ff;border-left:4px solid #3b82f6;padding:14px 18px;border-radius:8px;margin-bottom:20px;font-size:15px;}
.survey-section{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:18px 20px;margin-bottom:16px;}
.done-screen{text-align:center;padding:60px 20px;}
footer{display:none !important;}
"""

EMPTY_RESULTS = "<p style='color:#888;padding:20px;'>Resultaten verschijnen hier.</p>"

def make_progress(step, task_index=None):
    labels = ["1 Vooraf", "2 Setup", "3 Taken", "4 Enquête", "5 Klaar"]
    if step == 3 and task_index is not None:
        labels[2] = f"3 Taak {task_index+1}/5"
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
            f"<strong style='font-size:16px;'>Taak {t['id']}/5 — {t['title']}</strong><br>"
            f"<span style='color:#334155;'>{t['description']}</span></div>")

def mini_header_html(idx):
    t = TASKS[idx]
    return (f"<div style='background:#f0fdf4;border-left:4px solid #22c55e;"
            f"padding:14px 18px;border-radius:8px;margin-bottom:16px;'>"
            f"<strong>📝 Taak {t['id']}/5 afgerond — Korte check</strong><br>"
            f"<span style='color:#64748b;font-size:14px;'>{t['description']}</span></div>")

# ── Bouw de UI ────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Museumcollectie Onderzoek",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
    css=CUSTOM_CSS,
) as demo:

    session_state = gr.State({})
    chat_history  = gr.State([])

    gr.Markdown("# Museumcollectie Onderzoek")
    progress_bar = gr.HTML(make_progress(1))

    # ── Stap 1: Voorafgaande enquête ──────────────────────────────────────────
    with gr.Column(visible=True) as step1_col:
        gr.Markdown("## Stap 1 — Voordat we beginnen\nAlle antwoorden zijn anoniem en worden alleen voor onderzoeksdoeleinden gebruikt.")
        pre_pid = gr.Textbox(label="Deelnemers-ID (bijv. P01)", placeholder="P01")

        gr.Markdown("#### 👤 Achtergrond")
        with gr.Row():
            pre_age  = gr.Number(label="Leeftijd", minimum=18, maximum=99, value=None)
            pre_edu  = gr.Dropdown(label="Opleidingsniveau",
                choices=["Middelbaar onderwijs (HAVO/VWO/MBO)","Bachelor","Master","Doctoraat / PhD","Anders"])
            pre_lang = gr.Textbox(label="Moedertaal", placeholder="bijv. Nederlands")

        gr.Markdown("#### 🛠️ Ervaring met hulpmiddelen")

        gr.Markdown("**Hoe vertrouwd ben je met musea en/of kunstgeschiedenis?**  \n*1 = geen kennis · 5 = expert*")
        pre_museum = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        pre_ai = gr.Dropdown(label="Hoe vaak gebruik je AI-tools zoals ChatGPT?",
            choices=["Nooit","Zelden (een paar keer per jaar)","Soms (maandelijks)","Regelmatig (wekelijks)","Dagelijks"])

        gr.Markdown("**Hoe comfortabel ben je met het zoeken in databases of catalogi?**  \n*1 = helemaal niet · 5 = heel comfortabel*")
        pre_search = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        gr.Markdown("#### 🤖 Houding tegenover AI (AIAS-4)\n*1 = helemaal mee oneens · 5 = helemaal mee eens*")

        gr.Markdown("**AI-systemen presteren net zo goed als mensen.**")
        pre_a1 = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        gr.Markdown("**Ik vertrouw erop dat AI nauwkeurige informatie geeft.**")
        pre_a2 = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        gr.Markdown("**AI-tools zijn nuttig voor dagelijks werk.**")
        pre_a3 = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        gr.Markdown("**Ik ben comfortabel met het vertrouwen op AI voor informatie.**")
        pre_a4 = gr.Radio(["1","2","3","4","5"], label="", show_label=False)

        pre_btn = gr.Button("Opslaan & doorgaan →", variant="primary", size="lg")
        pre_out = gr.Markdown("")

    # ── Stap 2: Setup ─────────────────────────────────────────────────────────
    with gr.Column(visible=False) as step2_col:
        gr.Markdown("## Stap 2 — Sessievoorbereiding\n*In te vullen door de onderzoeker.*")
        with gr.Row():
            pid_box  = gr.Textbox(label="Deelnemers-ID", placeholder="P01")
            cond_box = gr.Dropdown(label="Conditie",
                choices=["A — Trefwoordzoeken","B — AI-chat"], value="A — Trefwoordzoeken")
        gr.HTML("<div style='background:#fefce8;border:1px solid #fbbf24;border-radius:8px;"
                "padding:12px 16px;margin:8px 0;font-size:14px;'>"
                "ℹ️ De deelnemer doorloopt <strong>alle 5 taken</strong> achtereenvolgens "
                "in de toegewezen conditie.</div>")
        setup_btn = gr.Button("Sessie starten →", variant="primary", size="lg")
        setup_out = gr.Markdown("")

    # ── Stap 3: Taakscherm ────────────────────────────────────────────────────
    with gr.Column(visible=False) as task_col:
        task_card = gr.HTML(task_card_html(0))

        with gr.Column(visible=True) as cond_a_col:
            gr.Markdown("### 🔍 Trefwoordzoeken\n*Resultaten zijn directe overeenkomsten — geen AI-interpretatie.*")
            with gr.Row():
                search_box = gr.Textbox(placeholder="bijv. ritueel object, 1700, lakwerk...",
                                        show_label=False, scale=8)
                search_btn = gr.Button("Zoeken", variant="primary", scale=1)
            search_results = gr.HTML(EMPTY_RESULTS)

        with gr.Column(visible=False) as cond_b_col:
            gr.Markdown("### 🤖 AI-onderzoeksassistent\n*AI-antwoorden kunnen fouten bevatten — verifieer altijd met bronlinks.*")
            chatbot  = gr.Chatbot(label="Chat", height=400)
            with gr.Row():
                chat_box = gr.Textbox(placeholder="Stel een vraag over de collectie...",
                                      show_label=False, scale=9)
                chat_btn = gr.Button("Versturen", variant="primary", scale=1)
            clear_btn = gr.Button("Chat wissen", size="sm")

        gr.Markdown("---")
        finish_btn = gr.Button("✅ Taak afronden →", variant="secondary", size="lg")
        finish_out = gr.Markdown("")

    # ── Stap 3b: Mini-enquête (per taak) ─────────────────────────────────────
    with gr.Column(visible=False) as mini_col:
        mini_header = gr.HTML(mini_header_html(0))

        mini_answer = gr.Textbox(
            label="📝 Wat is jouw antwoord op deze taak?",
            lines=3, placeholder="Schrijf hier je antwoord...")

        mini_completed = gr.Radio(
            label="✅ Heb je de taak afgerond?",
            choices=["✅ Ja", "⚠️ Gedeeltelijk", "❌ Nee"])

        gr.Markdown("**📊 Hoe zeker ben je van je antwoord?**  \n"
                    "*1 = helemaal niet zeker · 7 = volledig zeker*")
        mini_confidence = gr.Radio(
            choices=["1","2","3","4","5","6","7"],
            label="", show_label=False)

        mini_btn = gr.Button("Doorgaan →", variant="primary", size="lg")
        mini_out = gr.Markdown("")

    # ── Stap 4: Eindsurvey ────────────────────────────────────────────────────
    with gr.Column(visible=False) as final_col:
        gr.Markdown("## Stap 4 — Afsluitende enquête\n"
                    "Je hebt alle 5 taken afgerond! Beantwoord nog een paar vragen over je algehele ervaring.")
        survey_summary = gr.Markdown("")

        # Sectie: TOAST
        gr.Markdown("---\n### 🤝 Vertrouwen in het systeem (TOAST)")
        gr.Markdown("*1 = helemaal mee oneens · 7 = helemaal mee eens*")
        s_toast_r = gr.Radio(["1","2","3","4","5","6","7"],
                             label="Het systeem werkte betrouwbaar.")
        s_toast_c = gr.Radio(["1","2","3","4","5","6","7"],
                             label="Ik voelde me zeker bij het gebruik van dit systeem.")
        s_toast_t = gr.Radio(["1","2","3","4","5","6","7"],
                             label="Ik vond het systeem betrouwbaar.")

        # Sectie: NASA-TLX
        gr.Markdown("---\n### 🧠 Mentale inspanning (NASA-TLX)")
        gr.Markdown("*1 = helemaal niet · 7 = heel erg*")
        s_tlx_m = gr.Radio(["1","2","3","4","5","6","7"],
                           label="Hoeveel mentale inspanning kostte de sessie?")
        s_tlx_e = gr.Radio(["1","2","3","4","5","6","7"],
                           label="Hoe hard moest je werken tijdens de sessie?")

        # Sectie: SUS
        gr.Markdown("---\n### 💻 Gebruiksgemak (SUS)")
        gr.Markdown("*1 = helemaal mee oneens · 5 = helemaal mee eens*")
        s_sus1 = gr.Radio(["1","2","3","4","5"], label="Het systeem was gemakkelijk te gebruiken.")
        s_sus2 = gr.Radio(["1","2","3","4","5"], label="Ik voelde me zeker bij het gebruik van het systeem.")
        s_sus3 = gr.Radio(["1","2","3","4","5"], label="Ik zou dit systeem opnieuw willen gebruiken.")
        s_sus4 = gr.Radio(["1","2","3","4","5"], label="Het systeem gaf mij betrouwbare informatie.")
        s_sus5 = gr.Radio(["1","2","3","4","5"], label="Ik begreep waar de antwoorden vandaan kwamen.")

        # Sectie: Kritische evaluatie
        gr.Markdown("---\n### 🔍 Kritische evaluatie")
        s_verified = gr.Radio(
            label="Heb je antwoorden geverifieerd via de bronlinks?",
            choices=["✅ Ja, altijd","🔁 Soms","❌ Nee"])
        s_manip = gr.Radio(
            label="Wat voor hulpmiddel gebruikte je tijdens deze sessie?",
            choices=["Trefwoordzoeken","AI-assistent","Beide","Weet niet"])
        s_comments = gr.Textbox(
            label="💬 Opmerkingen of feedback? (optioneel)", lines=3,
            placeholder="Wat werkte goed? Wat was frustrerend?")

        gr.Markdown("---")
        final_err = gr.Markdown("")
        final_btn = gr.Button("Enquête versturen →", variant="primary", size="lg")

    # ── Stap 5: Klaar ─────────────────────────────────────────────────────────
    with gr.Column(visible=False) as done_col:
        gr.HTML("""
        <div class='done-screen'>
          <div style='font-size:64px;margin-bottom:16px;'>✅</div>
          <h2 style='font-size:26px;color:#166534;margin-bottom:8px;'>Hartelijk dank voor je deelname!</h2>
          <p style='color:#64748b;font-size:16px;'>De onderzoeker komt zo bij je.</p>
        </div>
        """)

    # ── Onderzoekersweergave ──────────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Accordion("🔒 Onderzoekersweergave", open=False):
        r_password = gr.Textbox(label="Wachtwoord", type="password", placeholder="Voer wachtwoord in")
        r_unlock   = gr.Button("Ontgrendelen", variant="primary")
        r_msg      = gr.Markdown("")
        r_log      = gr.Code(label="experiment_log.jsonl", language="json", lines=30, visible=False)
        with gr.Row(visible=False) as r_btn_row:
            r_refresh  = gr.Button("Vernieuwen", variant="secondary")
            r_download = gr.Button("⬇️ Log downloaden", variant="primary")
        r_file = gr.File(label="Download", visible=False)

    # ── Event handlers ─────────────────────────────────────────────────────────

    # Stap 1 → 2
    def on_pre_submit(pid, age, edu, lang, museum, ai_use, search_c, a1, a2, a3, a4, state):
        missing = []
        if not pid or not str(pid).strip():   missing.append("Deelnemers-ID")
        if age is None:                        missing.append("Leeftijd")
        if not edu:                            missing.append("Opleidingsniveau")
        if not lang or not str(lang).strip(): missing.append("Moedertaal")
        if museum is None:                    missing.append("Museum vertrouwdheid")
        if not ai_use:                        missing.append("AI-gebruik frequentie")
        if search_c is None:                  missing.append("Database comfort")
        if a1 is None:                        missing.append("AIAS vraag 1")
        if a2 is None:                        missing.append("AIAS vraag 2")
        if a3 is None:                        missing.append("AIAS vraag 3")
        if a4 is None:                        missing.append("AIAS vraag 4")
        if missing:
            return (f"⚠️ Vul alle velden in voor je verder gaat. Ontbreekt: {', '.join(missing)}",
                    gr.update(), gr.update(), gr.update())
        submit_pre_survey(pid, age, edu, lang, museum, ai_use, search_c, a1, a2, a3, a4, state)
        return (f"✅ Opgeslagen voor **{pid}**.",
                gr.update(value=make_progress(2)),
                gr.update(visible=False),
                gr.update(visible=True))

    pre_btn.click(
        on_pre_submit,
        [pre_pid, pre_age, pre_edu, pre_lang, pre_museum, pre_ai,
         pre_search, pre_a1, pre_a2, pre_a3, pre_a4, session_state],
        [pre_out, progress_bar, step1_col, step2_col],
    )

    # Stap 2 → Taak 1
    def on_start_session(pid, cond, state):
        if not pid.strip():
            return ("⚠️ Vul het Deelnemers-ID in.",
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
        is_a = cond.startswith("A")
        return (
            f"✅ Sessie gestart — {pid} · {'Conditie A (Trefwoordzoeken)' if is_a else 'Conditie B (AI-chat)'}",
            state,
            gr.update(value=make_progress(3, 0)),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=task_card_html(0)),
            gr.update(visible=is_a),
            gr.update(visible=not is_a),
        )

    setup_btn.click(
        on_start_session,
        [pid_box, cond_box, session_state],
        [setup_out, session_state, progress_bar,
         step2_col, task_col, task_card, cond_a_col, cond_b_col],
    )

    # Taak afronden → mini-enquête
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
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=mini_header_html(idx)),
        )

    finish_btn.click(
        on_finish_task,
        [session_state],
        [finish_out, session_state, progress_bar, task_col, mini_col, mini_header],
    )

    # Mini-enquête versturen → volgende taak of eindsurvey
    def on_mini_submit(answer, completed, confidence, state, history):
        NO_CHANGE = (gr.update(), gr.update(), gr.update(), gr.update(),
                     gr.update(), gr.update(), gr.update(), gr.update())
        if not answer or not answer.strip():
            return ("⚠️ Vul je antwoord in voor je doorgaat.",
                    state, history, *NO_CHANGE)
        if completed is None:
            return ("⚠️ Geef aan of je de taak hebt afgerond.",
                    state, history, *NO_CHANGE)
        if confidence is None:
            return ("⚠️ Geef je zekerheid aan (1–7).",
                    state, history, *NO_CHANGE)

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
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value=task_card_html(next_idx)),
                gr.update(value=""),
                gr.update(value=EMPTY_RESULTS),
                gr.update(value=[]),
                gr.update(),
            )
        else:
            log_event(pid, cond, "all_tasks_complete", {
                "total_queries": state.get("total_query_count", 0)
            })
            rows = "\n".join(
                f"| Taak {r['task_id']} | {r.get('completed','—')} | "
                f"Zekerheid: {r.get('confidence','—')}/7 | "
                f"{r.get('query_count',0)} zoekopdrachten | {r.get('elapsed_s',0)}s |"
                for r in state.get("task_results", [])
            )
            summary = (
                "**Jouw sessie in een oogopslag:**\n\n"
                "| Taak | Status | Zekerheid | Zoekopdrachten | Tijd |\n"
                "|------|--------|-----------|----------------|------|\n"
                + rows
            )
            return (
                "",
                state, history,
                gr.update(value=make_progress(4)),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(value=summary),
            )

    mini_btn.click(
        on_mini_submit,
        [mini_answer, mini_completed, mini_confidence, session_state, chat_history],
        [mini_out, session_state, chat_history, progress_bar,
         mini_col, task_col, final_col,
         task_card, search_box, search_results, chatbot, survey_summary],
    )

    # Eindsurvey → klaar
    def on_final_submit(toast_r, toast_c, toast_t, tlx_m, tlx_e,
                        sus1, sus2, sus3, sus4, sus5,
                        verified, manip, comments, state):
        required = [toast_r, toast_c, toast_t, tlx_m, tlx_e,
                    sus1, sus2, sus3, sus4, sus5, verified, manip]
        if any(v is None for v in required):
            return ("⚠️ Beantwoord alle vragen voor je de enquête verstuurt.",
                    gr.update(), gr.update(visible=True), gr.update(visible=False))
        submit_final_survey(toast_r, toast_c, toast_t, tlx_m, tlx_e,
                            sus1, sus2, sus3, sus4, sus5,
                            verified, manip, comments, state)
        return (
            "",
            gr.update(value=make_progress(5)),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    final_btn.click(
        on_final_submit,
        [s_toast_r, s_toast_c, s_toast_t, s_tlx_m, s_tlx_e,
         s_sus1, s_sus2, s_sus3, s_sus4, s_sus5,
         s_verified, s_manip, s_comments, session_state],
        [final_err, progress_bar, final_col, done_col],
    )

    # Conditie A
    search_btn.click(search_condition_a, [search_box, session_state], [search_results, session_state])
    search_box.submit(search_condition_a, [search_box, session_state], [search_results, session_state])

    # Conditie B
    chat_btn.click(chat_condition_b,
                   [chat_box, chat_history, session_state],
                   [chat_box, chatbot, session_state, chat_history])
    chat_box.submit(chat_condition_b,
                    [chat_box, chat_history, session_state],
                    [chat_box, chatbot, session_state, chat_history])
    clear_btn.click(lambda: ([], []), None, [chatbot, chat_history])

    # Onderzoekersweergave
    def unlock(pw):
        if pw == RESEARCHER_PASSWORD:
            return ("✅ Toegang verleend.",
                    gr.update(visible=True, value=load_log()),
                    gr.update(visible=True),
                    gr.update(visible=True))
        return ("❌ Onjuist wachtwoord.",
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

    r_unlock.click(unlock, [r_password], [r_msg, r_log, r_btn_row, r_file])
    r_refresh.click(load_log, None, r_log)
    r_download.click(download_log, None, r_file)

demo.launch()
