# Salesforce Case Copilot

A polished Streamlit demo that simulates an AI + Knowledge (RAG) workflow for Salesforce-like support cases. Upload a CSV of cases, add knowledge files, pick a case, and generate grounded triage JSON plus a draft reply in English or German.

## Features
- **Case ingestion**: upload a CSV with `case_id`, `subject`, and `description` fields.
- **Knowledge ingestion**: upload multiple `.txt` or `.md` files.
- **Retrieval (offline)**: TF-IDF vector search for evidence snippets.
- **Drafting**: LLM mode (if API key exists) or deterministic offline templates.
- **Reflection mode**: optional writer → critic → rewrite loop for improved clarity.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

Sample data is included in `sample_data/` and will load automatically if no files are uploaded.

## Environment variables (safe usage)
The app **never hardcodes keys** and only reads `OPENAI_API_KEY` from the environment.

**Recommended**: use a local `.env` file (already ignored by git).
```
OPENAI_API_KEY=your-key-here
```
Then export it in your shell:
```bash
export OPENAI_API_KEY="$(cat .env | cut -d= -f2-)"
```

If no key is provided, the app runs in **Offline mode** with deterministic templates.

## Repository structure
```
.
├── app.py
├── requirements.txt
├── sample_data/
│   ├── cases_demo.csv
│   └── knowledge_demo.md
└── .gitignore
```

## Notes
- Retrieval uses chunking (800 chars, 150 overlap) with filename + chunk id metadata.
- Responses explicitly mention when evidence is insufficient.
- No external services are required to run locally.
