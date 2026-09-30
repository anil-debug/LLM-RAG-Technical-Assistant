"""Streamlit view of one RAG request.

Run the API first. This process only calls HTTP. It does not load embedding
weights or an LLM itself.

    uv run uvicorn api.main:app --port 8000
    uv run streamlit run frontend/app.py
"""

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.view import citation_rows, hit_rows

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=120.0)


st.set_page_config(page_title="Technical RAG Assistant", layout="wide")
st.title("Technical RAG Assistant")
st.caption(
    "Upload a document, then ask a question. The page shows the rewritten query, "
    "hybrid retrieval scores, the context the model saw, and the citations that survived."
)

with st.sidebar:
    st.header("Documents")
    st.write(f"API: `{API_BASE}`")
    upload = st.file_uploader("Upload", type=["pdf", "md", "markdown", "txt", "html", "htm", "log"])
    if upload is not None and st.button("Index upload"):
        try:
            with _client() as client:
                response = client.post(
                    "/documents",
                    files={"file": (upload.name, upload.getvalue())},
                )
            if response.status_code >= 400:
                st.error(response.text)
            else:
                st.success(response.json())
        except httpx.HTTPError as exc:
            st.error(f"API unreachable: {exc}")
    if st.button("Refresh list"):
        st.session_state["refresh"] = True
    try:
        with _client() as client:
            listed = client.get("/documents")
            ready = client.get("/ready")
        if ready.status_code == 200:
            st.json(ready.json())
        if listed.status_code == 200:
            for document in listed.json():
                st.write(f"{document['filename']} — {document['chunk_count']} chunks")
    except httpx.HTTPError as exc:
        st.warning(f"Could not list documents: {exc}")

question = st.text_area("Question", placeholder="What does error 0x1F mean?")
if st.button("Ask", type="primary") and question.strip():
    history_id = st.session_state.get("conversation_id")
    try:
        with _client() as client:
            response = client.post(
                "/chat",
                json={"message": question.strip(), "conversation_id": history_id},
            )
    except httpx.HTTPError as exc:
        st.error(f"API unreachable: {exc}")
    else:
        if response.status_code >= 400:
            st.error(response.text)
        else:
            body = response.json()
            st.session_state["conversation_id"] = body["conversation_id"]
            st.subheader("Answer")
            st.write(body["answer"])
            timings = body.get("timings_ms") or {}
            st.write(
                "Latency ms: "
                + ", ".join(f"{key}={value}" for key, value in timings.items())
                + f" — model {body.get('model_name')}"
            )
            left, right = st.columns(2)
            with left:
                st.subheader("Rewritten query")
                st.code(body.get("rewritten_query") or "")
            with right:
                st.subheader("Citations")
                st.dataframe(citation_rows(body.get("citations") or []), use_container_width=True)
                rejected = body.get("rejected_citation_ids") or []
                if rejected:
                    st.warning(f"Rejected citation ids: {rejected}")
            st.subheader("Retrieved chunks")
            st.dataframe(hit_rows(body.get("hits") or []), use_container_width=True)
            st.subheader("Context sent to the model")
            st.code(body.get("context") or "")

if st.button("Clear conversation"):
    st.session_state.pop("conversation_id", None)
