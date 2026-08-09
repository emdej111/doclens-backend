"""DocLens Streamlit frontend.

Talks to the FastAPI backend over HTTP (BACKEND_URL). Run the backend first:
    uvicorn src.backend.api:app --reload
then:
    streamlit run src/frontend/app.py
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="DocLens", layout="wide")


def _init_state() -> None:
    st.session_state.setdefault("document_id", None)
    st.session_state.setdefault("document_meta", None)
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("truncated_for_summary", False)
    st.session_state.setdefault("chat_history", [])


def _reset_document_state() -> None:
    st.session_state.document_id = None
    st.session_state.document_meta = None
    st.session_state.summary = None
    st.session_state.truncated_for_summary = False
    st.session_state.chat_history = []


def _upload_document(uploaded_file) -> None:
    with st.spinner("Extracting text and generating summary..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.ConnectionError:
            st.error(
                f"Can't reach the backend at {BACKEND_URL}. "
                "Make sure it's running: `uvicorn src.backend.api:app --reload`"
            )
            return

    if response.status_code != 200:
        detail = _error_detail(response)
        st.error(f"Upload failed: {detail}")
        return

    data = response.json()
    st.session_state.document_id = data["document_id"]
    st.session_state.document_meta = {
        "filename": data["filename"],
        "num_pages": data["num_pages"],
        "num_chunks": data["num_chunks"],
    }
    st.session_state.summary = data["summary"]
    st.session_state.truncated_for_summary = data["truncated_for_summary"]
    st.session_state.chat_history = []


def _ask_question(question: str) -> None:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/ask",
            json={"document_id": st.session_state.document_id, "question": question},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach the backend at {BACKEND_URL}.")
        return

    if response.status_code != 200:
        detail = _error_detail(response)
        st.session_state.chat_history.append(
            {"question": question, "answer": f"Error: {detail}", "sources": []}
        )
        return

    data = response.json()
    st.session_state.chat_history.append(
        {"question": question, "answer": data["answer"], "sources": data["sources"]}
    )


def _error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def render_sidebar() -> None:
    with st.sidebar:
        st.title("DocLens")
        st.caption("AI document analyzer — upload a PDF, get a summary, ask questions.")

        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            is_new_file = (
                st.session_state.document_meta is None
                or st.session_state.document_meta["filename"] != uploaded_file.name
            )
            if is_new_file and st.button("Analyze document", type="primary"):
                _upload_document(uploaded_file)

        if st.session_state.document_meta:
            st.divider()
            meta = st.session_state.document_meta
            st.metric("Pages", meta["num_pages"])
            st.metric("Chunks indexed", meta["num_chunks"])
            if st.button("Clear document"):
                _reset_document_state()
                st.rerun()


def render_summary() -> None:
    st.subheader("Summary")
    if st.session_state.truncated_for_summary:
        st.info(
            "This document is long — the summary was generated from the first "
            f"{os.getenv('MAX_SUMMARY_CHARS', '60000')} characters rather than the full text."
        )
    st.write(st.session_state.summary)


def render_qa() -> None:
    st.subheader("Ask a question")

    question = st.chat_input("Ask something about the document...")
    if question:
        with st.spinner("Thinking..."):
            _ask_question(question)

    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["sources"]:
                with st.expander(f"Sources ({len(turn['sources'])} excerpts)"):
                    for source in turn["sources"]:
                        st.caption(f"Chunk #{source['chunk_index']} · similarity {source['score']}")
                        st.text(source["text"][:500] + ("..." if len(source["text"]) > 500 else ""))
                        st.divider()


def main() -> None:
    _init_state()
    render_sidebar()

    st.title("DocLens")

    if not st.session_state.document_id:
        st.info("Upload a PDF from the sidebar to get started.")
        return

    render_summary()
    st.divider()
    render_qa()


if __name__ == "__main__":
    main()
