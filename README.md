# Museum Search User Study

A Gradio web application built for a thesis user study comparing **keyword search** vs **AI-powered chat** for exploring a museum collection.

## What This Does

Participants are assigned to one of two conditions:

| Condition | Method | Description |
|-----------|--------|-------------|
| **A — Keyword Search** | Traditional | Text-based search through a CSV dataset of museum objects |
| **B — AI Chat** | RAG (Retrieval-Augmented Generation) | Conversational search powered by Gemini 2.5 Flash with a file search index |

Participants complete 5 tasks (finding ritual objects, oldest object, lacquerwork, objects from ~1700, anonymous makers) and their interactions are logged for research analysis.

## Tech Stack

- **Frontend:** [Gradio](https://gradio.app) 6.14 (Python 3.13)
- **AI Model:** Google Gemini 2.5 Flash via `google-genai`
- **Dataset:** `small_dataset.csv` — museum collection records (ID, Title, Type, Creator, Creation Date, Image URL)
- **Logging:** JSONL file (`experiment_log.jsonl`) per session

## Project Structure

```
├── app.py                  # Main application
├── small_dataset.csv       # Museum collection dataset
├── requirement.txt         # Python dependencies
└── experiment_log.jsonl    # Generated at runtime (not tracked)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `STORE_NAME` | No | Gemini file search store name (defaults to study store) |
| `RESEARCHER_PASSWORD` | No | Password for researcher dashboard (default: `mypassword123`) |

## Running Locally

```bash
pip install gradio google-genai pandas
export GEMINI_API_KEY=your_key_here
python app.py
```

## Live Demo

Deployed on Hugging Face Spaces: [flugger/aryen](https://huggingface.co/spaces/flugger/aryen)

## Research Context

This application is part of a thesis investigating how AI-assisted search interfaces affect user experience and task performance in digital museum collections.
