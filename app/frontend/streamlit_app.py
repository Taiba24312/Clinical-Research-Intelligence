"""
Clinical Research Intelligence Assistant - Streamlit demo.

Query -> parsed filters -> top-10 trials (BM25 + dense + RRF) -> per-trial
verification of the top-5 (RAG v2: match badge + evidence quote) ->
grounded answer with NCT citations from a local LLM.

Run from the project root (conda env clinical_rag_clean):
    streamlit run app/frontend/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import FINAL_PIPELINE  # noqa: E402
from src.generation.context_builder import TOP_K_CONTEXT  # noqa: E402
from src.generation.llm import DEFAULT_MODEL, LocalLLM, answer_query  # noqa: E402
from src.generation.rag_v2 import DEFAULT_PROMPT_VERSION, RAGv2  # noqa: E402
from src.retrieval.pipeline import TrialSearchEngine  # noqa: E402


TOP_K_RESULTS = 10
EXAMPLES = [
    "Recruiting NSCLC trials involving pembrolizumab",
    "Phase 3 breast cancer trials",
    "Parkinson's disease trials that are not yet recruiting",
    "Observational studies for cardiovascular disease",
    "Pediatric asthma trials",
]
BADGES = {2: "✅ MATCH", 1: "🟡 PARTIAL MATCH", 0: "⚪ NO MATCH"}

st.set_page_config(page_title="Clinical Trial Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading search indexes (first run only)...")
def get_engine() -> TrialSearchEngine:
    return TrialSearchEngine(**FINAL_PIPELINE)


@st.cache_resource(show_spinner="Loading the local language model (first run only)...")
def get_llm(device: str) -> LocalLLM:
    return LocalLLM(device=device)


@st.cache_resource(show_spinner=False)
def get_rag_v2(device: str) -> RAGv2:
    return RAGv2(get_llm(device))


def pretty(value: str) -> str:
    return str(value or "NA").replace("_", " ").replace("; ", " / ").title()


def escape(text: str) -> str:
    return str(text).replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")


st.title("🔎 Clinical Research Intelligence Assistant")
st.caption("Searches a frozen snapshot of 4,695 ClinicalTrials.gov records (data version 2026-10-07). "
           "Research demo — not medical advice; always check the trial page.")

with st.sidebar:
    st.header("Options")
    generate = st.checkbox("Verify trials and generate an answer (local LLM)", value=True)
    mode = st.radio("Answer method", [f"RAG v2 - verify each trial ({DEFAULT_PROMPT_VERSION})",
                                      "RAG v1 - single prompt"], index=0)
    st.markdown(f"**Model:** `{DEFAULT_MODEL}`  \n**Retrieval:** BM25 (tokenizer "
                f"{FINAL_PIPELINE['tokenizer']}) + bge-small, RRF  \n"
                f"**Parser:** {FINAL_PIPELINE['parser_version']} (rule-based)  \n"
                f"Configuration frozen in `src/config.py`.")
    st.markdown("**Example queries**")
    for example in EXAMPLES:
        st.code(example, language=None)

query = st.text_input("Ask about clinical trials", placeholder=EXAMPLES[0])

if query.strip():
    engine = get_engine()
    result = engine.search(query.strip(), top_k=TOP_K_RESULTS)
    use_v2 = generate and mode.startswith("RAG v2") and bool(result.trials)

    st.subheader("1. How the query was understood")
    left, right = st.columns(2)
    with left:
        st.markdown("**Structured filters** (applied before ranking)")
        st.json(result.filters or {"(none)": "no structured filter recognized"})
    with right:
        st.markdown("**Text used for ranking**")
        st.code(result.semantic_query, language=None)
        st.markdown(f"**Trials passing the filters:** {result.eligible_trials:,}")

    v2_output = None
    if use_v2:
        with st.spinner(f"Verifying the top {TOP_K_CONTEXT} trials and writing the answer..."):
            v2_output = get_rag_v2(engine.device).run(query.strip(), result.trials[:TOP_K_CONTEXT])
        concepts = v2_output["concepts"]
        with right:
            st.markdown(f"**Concepts checked per trial:** disease = `{concepts['disease'] or '—'}`, "
                        f"intervention = `{concepts['intervention'] or '—'}`")
    verification = {v["nct_id"]: v for v in (v2_output or {}).get("verifications", [])}

    st.subheader(f"2. Top {len(result.trials)} trials")
    if not result.trials:
        st.warning("No trial in the corpus satisfies these filters.")
    for trial in result.trials:
        with st.container(border=True):
            st.markdown(f"**{trial['rank']}. [{escape(trial['title'])}]({trial['url']})**")
            st.caption(f"{trial['nct_id']} · {pretty(trial['status'])} · Phase: {pretty(trial['phase'])} · "
                       f"{pretty(trial['study_type'])}")
            st.markdown(f"**Conditions:** {escape(trial['conditions'])}  \n"
                        f"**Interventions:** {escape(trial['interventions'])}")
            v = verification.get(trial["nct_id"])
            if v is not None:
                note = ("" if v["quote_found"] else
                        " — ⚠️ evidence quote not found in the trial text, verification dropped")
                st.markdown(f"**{BADGES[v['verified_grade']]}** · disease: `{v['disease_match']}` · "
                            f"intervention: `{v['intervention_match']}`{note}")
                if v["evidence"]:
                    st.caption(f"Evidence: “{escape(v['evidence'])}”")
            elif use_v2:
                st.caption(f"Not verified (only the top {TOP_K_CONTEXT} trials are checked).")

    st.subheader("3. Answer")
    if not generate:
        st.info("Verification and answer generation are switched off in the sidebar.")
    elif not result.trials:
        st.info("No trials retrieved, so there is nothing to answer from.")
    elif v2_output is not None:
        st.markdown(v2_output["answer"])
        st.caption(f"RAG v2: answered only from trials verified among the top {TOP_K_CONTEXT} · "
                   f"{len(v2_output['cited'])} trial(s) cited · {v2_output['latency_s']} s")
        if v2_output["invalid_citations"] or v2_output["cited_unverified"]:
            st.error("The answer cites trials that were not verified: "
                     f"{v2_output['invalid_citations'] + v2_output['cited_unverified']}")
    else:
        llm = get_llm(engine.device)
        with st.spinner("Generating a grounded answer..."):
            output = answer_query(query.strip(), result.trials, llm)
        st.markdown(output["answer"])
        st.caption(f"RAG v1: answered only from the top {TOP_K_CONTEXT} trials above · "
                   f"{len(output['cited'])} trial(s) cited · {output['latency_s']} s")
        if output["invalid_citations"]:
            st.error(f"The answer cites trials that were not in its context: {output['invalid_citations']}")
