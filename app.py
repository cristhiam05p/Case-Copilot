import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SAMPLE_CASES_PATH = "sample_data/cases_demo.csv"
SAMPLE_KNOWLEDGE_PATH = "sample_data/knowledge_demo.md"


@st.cache_data(show_spinner=False)
def load_cases(uploaded_file: Optional[Any]) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.read_csv(SAMPLE_CASES_PATH)
    return pd.read_csv(uploaded_file)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(end - overlap, 0)
    return chunks


@st.cache_data(show_spinner=False)
def load_knowledge(
    file_payloads: Tuple[Tuple[str, bytes], ...],
) -> List[Dict[str, Any]]:
    sources: List[Tuple[str, str]] = []
    if file_payloads:
        for filename, data in file_payloads:
            sources.append((filename, data.decode("utf-8", errors="ignore")))
    else:
        with open(SAMPLE_KNOWLEDGE_PATH, "r", encoding="utf-8") as handle:
            sources.append((os.path.basename(SAMPLE_KNOWLEDGE_PATH), handle.read()))

    chunks: List[Dict[str, Any]] = []
    for filename, text in sources:
        for idx, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "id": f"{filename}-{idx}",
                "filename": filename,
                "chunk_index": idx,
                "text": chunk,
            })
    return chunks


@st.cache_resource(show_spinner=False)
def build_retriever(texts: Tuple[str, ...]) -> Tuple[Optional[TfidfVectorizer], Optional[Any]]:
    if not texts:
        return None, None
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_evidence(
    query: str,
    chunks: List[Dict[str, Any]],
    vectorizer: Optional[TfidfVectorizer],
    matrix: Optional[Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not chunks or vectorizer is None or matrix is None:
        return []
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]
    evidence = []
    for idx in ranked_indices:
        item = chunks[idx].copy()
        item["score"] = float(scores[idx])
        evidence.append(item)
    return evidence


def _infer_urgency(text: str) -> str:
    lowered = text.lower()
    high_signals = ["urgent", "outage", "down", "blocked", "security", "payment failed"]
    medium_signals = ["issue", "error", "delay", "slow", "unable", "problem"]
    if any(token in lowered for token in high_signals):
        return "high"
    if any(token in lowered for token in medium_signals):
        return "medium"
    return "low"


def generate_triage(
    case_row: pd.Series,
    evidence: List[Dict[str, Any]],
    language: str,
    llm_enabled: bool,
    client: Optional[OpenAI],
) -> Dict[str, Any]:
    if llm_enabled and client is not None:
        evidence_text = "\n".join(
            f"[{idx + 1}] {item['filename']}#{item['chunk_index']}: {item['text']}"
            for idx, item in enumerate(evidence)
        )
        prompt = (
            "You are a support triage assistant for Salesforce-like cases. "
            "Only use the provided evidence. If evidence is insufficient, say so explicitly. "
            "Return JSON only with keys: category, urgency, summary, next_action. "
            "Summary must be one sentence.\n\n"
            f"Case subject: {case_row['subject']}\n"
            f"Case description: {case_row['description']}\n\n"
            f"Evidence:\n{evidence_text if evidence_text else 'No evidence provided.'}\n"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    urgency = _infer_urgency(f"{case_row['subject']} {case_row['description']}")
    category = "Technical Issue" if urgency != "low" else "General Inquiry"
    summary = " ".join(str(case_row["description"]).split()[:18])
    next_action = "Collect logs and confirm reproduction steps."
    if language == "DE":
        next_action = "Logs sammeln und Reproduktionsschritte bestätigen."
    return {
        "category": category,
        "urgency": urgency,
        "summary": summary,
        "next_action": next_action,
    }


def _offline_reply_template(
    case_row: pd.Series,
    evidence: List[Dict[str, Any]],
    language: str,
) -> str:
    if language == "DE":
        greeting = "Hallo"
        closing = "Viele Grüße\nIhr Support-Team"
        insuff = "Hinweis: Die verfügbaren Wissensartikel reichen für eine vollständige Antwort nicht aus."
    else:
        greeting = "Hello"
        closing = "Best regards\nSupport Team"
        insuff = "Note: The available knowledge articles are insufficient for a complete answer."

    evidence_note = ""
    if not evidence:
        evidence_note = f"\n\n{insuff}"

    summary = " ".join(str(case_row["description"]).split()[:25])
    steps = "Please confirm the exact error message and share recent logs or screenshots."
    if language == "DE":
        steps = "Bitte bestätigen Sie die genaue Fehlermeldung und senden Sie aktuelle Logs oder Screenshots."

    return (
        f"{greeting},\n\n"
        f"vielen Dank für Ihre Anfrage zu '{case_row['subject']}'. "
        f"Zusammenfassung: {summary}.\n\n"
        f"Nächste Schritte: {steps}\n"
        f"{evidence_note}\n\n"
        f"{closing}"
        if language == "DE"
        else (
            f"{greeting},\n\n"
            f"thanks for reaching out about '{case_row['subject']}'. "
            f"Summary: {summary}.\n\n"
            f"Next steps: {steps}\n"
            f"{evidence_note}\n\n"
            f"{closing}"
        )


def generate_reply(
    case_row: pd.Series,
    evidence: List[Dict[str, Any]],
    language: str,
    llm_enabled: bool,
    client: Optional[OpenAI],
) -> str:
    if llm_enabled and client is not None:
        evidence_text = "\n".join(
            f"[{idx + 1}] {item['filename']}#{item['chunk_index']}: {item['text']}"
            for idx, item in enumerate(evidence)
        )
        prompt = (
            "You draft a customer reply for a Salesforce-like case. "
            "Only use the provided evidence. If evidence is insufficient, say so explicitly. "
            f"Reply in {language}. Use a professional, concise tone.\n\n"
            f"Case subject: {case_row['subject']}\n"
            f"Case description: {case_row['description']}\n\n"
            f"Evidence:\n{evidence_text if evidence_text else 'No evidence provided.'}\n"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or _offline_reply_template(
            case_row, evidence, language
        )

    return _offline_reply_template(case_row, evidence, language)


def reflect_and_improve(
    reply: str,
    evidence: List[Dict[str, Any]],
    language: str,
    client: Optional[OpenAI],
) -> str:
    if client is None:
        return reply
    evidence_text = "\n".join(
        f"[{idx + 1}] {item['filename']}#{item['chunk_index']}: {item['text']}"
        for idx, item in enumerate(evidence)
    )
    critic_prompt = (
        "You are a critic reviewing a support reply. Evaluate for clarity, corporate tone, "
        "groundedness to evidence, and policy compliance. Provide concise bullet feedback.\n\n"
        f"Evidence:\n{evidence_text if evidence_text else 'No evidence provided.'}\n\n"
        f"Reply ({language}):\n{reply}\n"
    )
    critique = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": critic_prompt}],
    ).choices[0].message.content

    rewrite_prompt = (
        "You are a writer improving a support reply. Apply the critic feedback, "
        "stay grounded to evidence, and keep it concise. If evidence is insufficient, say so.\n\n"
        f"Critic feedback:\n{critique}\n\n"
        f"Original reply ({language}):\n{reply}\n"
    )
    improved = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[{"role": "user", "content": rewrite_prompt}],
    ).choices[0].message.content

    return improved or reply


def main() -> None:
    st.set_page_config(page_title="Salesforce Case Copilot", layout="wide")
    st.title("Salesforce Case Copilot")
    st.write(
        "Simulate an AI + Knowledge integration to triage cases and draft replies. "
        "Upload cases and knowledge files, select a case, and generate a grounded draft."
    )

    api_key = os.getenv("OPENAI_API_KEY")
    llm_enabled = bool(api_key)
    client = OpenAI(api_key=api_key) if llm_enabled else None

    with st.sidebar:
        st.header("Inputs")
        cases_file = st.file_uploader("Upload cases CSV", type=["csv"])
        knowledge_files = st.file_uploader(
            "Upload knowledge files (txt/md)",
            type=["txt", "md"],
            accept_multiple_files=True,
        )
        language = st.selectbox("Language", options=["EN", "DE"], index=0)
        top_k = st.slider("Top-k evidence", min_value=1, max_value=5, value=3)
        reflection_enabled = st.toggle("Reflection mode", value=False)

        if llm_enabled:
            st.success("LLM mode enabled")
        else:
            st.info("Offline mode (no API key found)")

    cases_df = load_cases(cases_file)
    st.subheader("Case preview")
    st.dataframe(cases_df, use_container_width=True)

    case_ids = cases_df["case_id"].astype(str).tolist()
    selected_case_id = st.selectbox("Select a case_id", options=case_ids)

    file_payloads = tuple(
        (file.name, file.getvalue()) for file in knowledge_files
    )
    knowledge_chunks = load_knowledge(file_payloads)
    texts = tuple(chunk["text"] for chunk in knowledge_chunks)
    vectorizer, matrix = build_retriever(texts)

    if st.button("Generate Draft"):
        case_row = cases_df.loc[cases_df["case_id"].astype(str) == selected_case_id].iloc[0]
        query = f"{case_row['subject']} {case_row['description']}"
        evidence = retrieve_evidence(query, knowledge_chunks, vectorizer, matrix, top_k)

        triage = generate_triage(case_row, evidence, language, llm_enabled, client)
        reply = generate_reply(case_row, evidence, language, llm_enabled, client)

        if reflection_enabled and llm_enabled:
            reply = reflect_and_improve(reply, evidence, language, client)

        st.subheader("Triage JSON")
        st.code(json.dumps(triage, indent=2, ensure_ascii=False), language="json")

        st.subheader("Draft reply")
        st.markdown(reply)

        st.subheader("Evidence")
        if not evidence:
            st.write("No evidence retrieved.")
        for item in evidence:
            with st.expander(f"{item['filename']} · chunk {item['chunk_index']} · score {item['score']:.2f}"):
                st.write(item["text"])


if __name__ == "__main__":
    main()
