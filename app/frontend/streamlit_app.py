"""
Clinical Research Intelligence Assistant - Streamlit demo.

Query -> parsed filters -> top-10 trials (BM25 + dense + RRF) -> grounded
answer from a local LLM with NCT citations.

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
from src.retrieval.pipeline import TrialSearchEngine  # noqa: E402


TOP_K_RESULTS = 10
EXAMPLES = [
    "Recruiting NSCLC trials involving pembrolizumab",
    "Phase 3 breast cancer trials",
    "Parkinson's disease trials that are not yet recruiting",
    "Observational studies for cardiovascular disease",
    "Pediatric asthma trials",
]

st.set_page_config(page_title="Clinical Trial Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading search indexes (first run only)...")
def get_engine() -> TrialSearchEngine:
    return TrialSearchEngine(**FINAL_PIPELINE)


@st.cache_resource(show_spinner="Loading the local language model (first run only)...")
def get_llm(device: str) -> LocalLLM:
    return LocalLLM(device=device)


def pretty(value: str) -> str:
    return str(value or "NA").replace("_", " ").replace("; ", " / ").title()


def escape(text: str) -> str:
    return str(text).replace("[", "\\[").replace("]", "\\]").replace("*", "\\*")


st.title("🔎 Clinical Research Intelligence Assistant")
st.caption("Searches a frozen snapshot of 4,695 ClinicalTrials.gov records (data version 2026-10-07). "
           "Research demo — not medical advice; always check the trial page.")

with st.sidebar:
    st.header("Options")
    generate = st.checkbox("Generate an answer with the local LLM", value=True)
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

    st.subheader("1. How the query was understood")
    left, right = st.columns(2)
    with left:
        st.markdown("**Structured filters** (applied before ranking)")
        st.json(result.filters or {"(none)": "no structured filter recognized"})
    with right:
        st.markdown("**Text used for ranking**")
        st.code(result.semantic_query, language=None)
        st.markdown(f"**Trials passing the filters:** {result.eligible_trials:,}")

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

    st.subheader("3. Answer")
    if not generate:
        st.info("Answer generation is switched off in the sidebar.")
    elif not result.trials:
        st.info("No trials retrieved, so there is nothing to answer from.")
    else:
        llm = get_llm(engine.device)
        with st.spinner("Generating a grounded answer..."):
            output = answer_query(query.strip(), result.trials, llm)
        st.markdown(output["answer"])
        st.caption(f"Answered only from the top {TOP_K_CONTEXT} trials above · "
                   f"{len(output['cited'])} trial(s) cited · {output['latency_s']} s")
        if output["invalid_citations"]:
            st.error(f"The answer cites trials that were not in its context: {output['invalid_citations']}")
