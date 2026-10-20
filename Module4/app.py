"""
Streamlit UI for the RAG Document Chatbot with Hallucination Detection.
Run from the project folder:  streamlit run app.py --server.fileWatcherType none
Works with both layouts: modules/ folder, or Module1..Module4 folders.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = HERE if os.path.isdir(os.path.join(HERE, "modules")) else os.path.dirname(HERE)
for path in (HERE, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st

try:
    from modules.document_processor import process_document
    from modules.retriever import SemanticRetriever
    from modules.answer_generator import generate_answer, STATUS_SUCCESS, STATUS_INSUFFICIENT
    from modules.hallucination_detector import HallucinationDetector
except ModuleNotFoundError:
    from Module1.document_processor import process_document
    from Module2.retriever import SemanticRetriever
    from Module3.answer_generator import generate_answer, STATUS_SUCCESS, STATUS_INSUFFICIENT
    from hallucination_detector import HallucinationDetector

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
TOP_K = 5   # passages retrieved for every question (fixed; change here if needed)

STATUS_LABELS = {
    "supported": "✅ Confirmed by the document",
    "contradicted": "❌ Conflicts with the document",
    "not_supported": "⚠️ Not confirmed by the document",
    "conflicting_evidence": "🔀 The document gives mixed signals",
}
NO_EVIDENCE_REASONS = {
    "insufficient_evidence": "No evidence passages were available.",
    "no_relevant_sentence": "No sentence in the retrieved passages was similar enough to this statement.",
}

st.set_page_config(page_title="Delirium", page_icon="🌀", layout="centered")


@st.cache_resource(show_spinner="Loading the fact-checking model (only needed once per session)...")
def get_detector() -> HallucinationDetector:
    """One detector per app process. The NLI model is loaded here, at startup,
    so the first question is not slowed down by it."""
    detector = HallucinationDetector()      # e.g. HallucinationDetector(model_name="...") to use another model
    detector.load_model()
    return detector


def save_uploaded_file(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
STATUS_COLORS = {          # RGB values for the answer highlights
    "supported": "46,160,67",
    "contradicted": "218,54,51",
    "not_supported": "227,160,8",
    "conflicting_evidence": "130,80,223",
}


def build_highlighted_html(answer: str, claims: list):
    """Wrap each checked statement in a coloured span. Returns (markdown, highlighted?)."""
    esc = lambda t: t.replace("<", "&lt;")      # never let answer text inject HTML tags
    parts, cursor, highlighted = [], 0, False
    for c in claims:
        idx = answer.find(c["claim"], cursor)    # claims appear in answer order
        if idx < 0:
            continue                             # e.g. markdown was stripped: skip highlight
        rgb = STATUS_COLORS.get(c["status"], "128,128,128")
        label = c["status"].replace("_", " ")
        parts.append(esc(answer[cursor:idx]))
        parts.append(f'<span title="{label}" style="background-color: rgba({rgb},0.25); '
                     f'border-radius: 4px; padding: 1px 3px;">{esc(c["claim"])}</span>')
        cursor = idx + len(c["claim"])
        highlighted = True
    parts.append(esc(answer[cursor:]))
    return "".join(parts), highlighted


def render_answer(answer: str, report) -> None:
    """Answer card with each statement colour-coded by its verification status."""
    claims = [] if (not report or report.get("error")) else report["claims"]
    body, highlighted = build_highlighted_html(answer, claims)
    with st.container(border=True):
        st.markdown(body, unsafe_allow_html=True)
    if highlighted:
        st.caption("Highlights: 🟩 confirmed · 🟨 not confirmed · 🟥 conflicts · 🟪 mixed "
                   "(hover a highlight to see its status)")
    with st.expander("📋 Copy as plain text"):
        st.code(answer, language=None)


def describe_evidence(e: dict):
    """Turn the model's scores into a plain-language verdict and topic-match label."""
    confirms, opposes = e["entailment_score"], e["contradiction_score"]
    sim = e.get("similarity", 0)
    relevance = "very relevant" if sim >= 0.75 else "relevant" if sim >= 0.5 else "loosely related"
    if e.get("list_item") and confirms < 0.5:
        # One entry in a bullet list names one item; it says nothing about the other items.
        return "⚪ This is a different entry in a list, so it does not confirm the statement.", relevance
    if opposes >= 0.8:
        verdict = "❌ This passage says the opposite."
    elif confirms >= 0.8:
        verdict = "✅ This passage clearly confirms the statement."
    elif confirms >= 0.5:
        verdict = "🟡 This passage partly confirms it, but not clearly enough."
    elif opposes >= 0.5:
        verdict = "🟠 This passage may conflict with the statement."
    else:
        verdict = "⚪ This passage is related but does not confirm the statement."
    return verdict, relevance


def render_sources(sources: list) -> None:
    """Show 'Sources: file — p.4, p.5' grouped by file."""
    pages_by_file = {}
    for s in sources:
        pages = pages_by_file.setdefault(s["source"], [])
        if s["page"] not in pages:
            pages.append(s["page"])
    if pages_by_file:
        text = "  |  ".join(f"{name} — " + ", ".join(f"p.{p}" for p in pages)
                            for name, pages in pages_by_file.items())
        st.caption(f"📄 Sources: {text}")


def render_verification(report: dict) -> None:
    """Verdict banner, metrics, and per-claim details."""
    st.markdown("##### 🔎 Fact-check against your document")
    if report["error"]:
        st.error(f"Verification failed: {report['error']}")
        return

    s = report["summary"]
    total, supported = s["total_claims"], s["supported"]
    if total == 0:
        st.info("No factual statements could be checked in this answer.")
        return

    if s["overall_status"] == "verified":
        st.success(f"All {total} statements are supported by your document.")
    elif s["overall_status"] == "partially_verified":
        st.warning(f"{supported} of {total} statements are supported. Review the others below.")
    else:
        st.error(f"None of the {total} statements could be confirmed from your document.")

    st.progress(supported / total)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed", supported)
    c2.metric("Conflicts", s["contradicted"])
    c3.metric("Not confirmed", s["not_supported"])
    c4.metric("Mixed", s["conflicting_evidence"])
    st.caption("This is an automated estimate, not a guarantee. 'Not confirmed' means the document "
               "does not confirm the statement; it does not mean the statement is false.")

    for i, c in enumerate(report["claims"], start=1):
        label = STATUS_LABELS.get(c["status"], c["status"])
        if c["status"] == "not_supported" and any(e["entailment_score"] >= 0.5 for e in c["evidence"]):
            label = "🟡 Only partly confirmed by the document"
        with st.expander(f"{label} — {c['claim']}"):
            if c.get("compound"):
                st.caption("This sentence may contain several statements; it was checked as one.")
            if c.get("ignored_conflicts"):
                st.caption("A less relevant passage seemed to disagree and was set aside; "
                           "the most closely matching passage was used.")
            if not c["evidence"]:
                st.write(NO_EVIDENCE_REASONS.get(c.get("reason"), "No evidence found."))
            # Best-matching evidence first
            for e in sorted(c["evidence"], key=lambda e: -e.get("similarity", 0)):
                st.markdown(f"**📄 {e['source']} — page {e['page']}**")
                st.markdown(f"> {e['text']}")
                verdict, relevance = describe_evidence(e)
                st.markdown(f"{verdict}  \n*Topic match: {relevance}*")
                confirms = min(max(e["entailment_score"], 0.0), 1.0)
                st.progress(confirms, text=f"Confirms the statement: {confirms:.0%}")
                if e.get("truncated"):
                    st.caption("This passage was long, so only its first part was checked.")
                if st.session_state.get("show_technical"):
                    st.caption(f"Technical scores: match {e['similarity']:.2f} · "
                               f"supports {e['entailment_score']:.2f} · "
                               f"contradicts {e['contradiction_score']:.2f} · "
                               f"unrelated {e['neutral_score']:.2f}")


def render_assistant(msg: dict) -> None:
    result = msg["result"]
    ok = result["status"] in (STATUS_SUCCESS, STATUS_INSUFFICIENT)
    if not ok:
        st.error(f"{result['answer']}  (status: {result['status']})")
        return
    if result["status"] == STATUS_INSUFFICIENT:
        st.info("ℹ️ " + result["answer"])        # the document does not contain enough information
    else:
        render_answer(result["answer"], msg["report"])
    render_sources(result["sources"])
    render_verification(msg["report"])
    with st.expander("📚 Retrieved passages used as evidence"):
        for i, r in enumerate(msg["retrieved"], start=1):
            st.markdown(f"**{i}. Page {r['page']}** · retrieval score {r['score']:.2f}")
            st.write(r["text"])


# ---------------------------------------------------------------------------
# Sidebar: document setup
# ---------------------------------------------------------------------------
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("doc_name", None)
st.session_state.setdefault("stats", (0, 0))
st.session_state.setdefault("history", [])

with st.sidebar:
    st.header("📄 Your document")
    uploaded_file = st.file_uploader("1. Upload a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("2. Process document", use_container_width=True,
                                               type="primary"):
        pdf_path = save_uploaded_file(uploaded_file)
        try:
            with st.status("Processing document...", expanded=True) as status:
                st.write("Reading and chunking the PDF...")
                chunks = process_document(pdf_path)
                st.write("Building the search index (first run downloads a model)...")
                retriever = SemanticRetriever()
                retriever.build_index(chunks)
                status.update(label="Document ready", state="complete", expanded=False)
            st.session_state.retriever = retriever
            st.session_state.doc_name = uploaded_file.name
            st.session_state.stats = (len({c["page"] for c in chunks}), len(chunks))
            st.session_state.history = []        # new document -> fresh conversation
        except Exception as e:
            st.error(f"Could not process document: {e}")

    if st.session_state.retriever is not None:
        pages, num_chunks = st.session_state.stats
        st.success(f"Ready: {st.session_state.doc_name}")
        m1, m2 = st.columns(2)
        m1.metric("Pages", pages)
        m2.metric("Chunks", num_chunks)

    st.divider()
    st.checkbox("Show technical scores", key="show_technical",
                help="Adds the raw model scores under each piece of evidence.")
    with st.expander("ℹ️ How to read the results"):
        st.markdown(
            "- **Highlights** in the answer show how well the document backs each sentence.\n"
            "- **Confirmed**: a passage in your document says the same thing.\n"
            "- **Not confirmed**: the document does not clearly say this. It may still be true.\n"
            "- **Conflicts**: a passage says something different.\n"
            "- **Topic match** shows how closely a passage relates to the statement.\n"
            "- This is an automated check and can be wrong. Open the evidence to verify.")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area: chat
# ---------------------------------------------------------------------------
st.title("🌀Delirium")
st.caption("Ask questions about your PDF. Every answer is fact-checked against the document.")

try:
    get_detector()      # preload the fact-checking model now; cached for the rest of the session
except Exception as e:
    st.warning(f"The fact-checking model could not be loaded yet: {e}")

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process document** to begin.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant(msg)

question = st.chat_input("Ask a question about your document",
                         disabled=st.session_state.retriever is None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                retrieved = st.session_state.retriever.search(question, top_k=TOP_K)
            with st.spinner("Writing the answer..."):
                result = generate_answer(question, retrieved)
            report = None
            if result["status"] in (STATUS_SUCCESS, STATUS_INSUFFICIENT):
                with st.spinner("Fact-checking the answer..."):
                    detector = get_detector()
                    detector.embedder = st.session_state.retriever.model
                    report = detector.verify_answer(question, result["answer"], retrieved)
            msg = {"role": "assistant", "result": result, "report": report, "retrieved": retrieved}
            render_assistant(msg)
            st.session_state.history.append(msg)
        except Exception as e:
            st.error(f"Something went wrong: {e}")