import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set single-threaded OpenMP env vars before any torch/lightgbm import (via the
# agent/rag imports below) to avoid a macOS libomp segfault. See agent/runtime.py.
import agent.runtime  # noqa: F401,E402

import streamlit as st

from agent.run import ask_dynasty_scout
from rag.retrieval_tool import get_vector_store_stats

st.set_page_config(page_title="Dynasty Scout AI", page_icon="🏈", layout="centered")

st.title("🏈 Dynasty Scout AI")
st.caption(
    "ML-powered dynasty fantasy football analysis — LangGraph agent over "
    "LightGBM forecasts, calibrated injury risk, and RAG-retrieved scouting context."
)

with st.sidebar:
    st.subheader("Vector store status")
    try:
        stats = get_vector_store_stats()
        st.metric("Indexed chunks", stats.get("total_chunks", 0))
        if stats.get("source_type_distribution"):
            st.write("**By source:**")
            for src, count in stats["source_type_distribution"].items():
                st.write(f"- {src.replace('_', ' ').title()}: {count}")
    except Exception as e:
        st.warning(f"Could not load vector store stats: {e}")

    st.divider()
    st.subheader("Example queries")
    examples = [
        "Should I trade for Justin Jefferson?",
        "What's the injury risk for Saquon Barkley next season?",
        "Evaluate this rookie WR for my dynasty rebuild",
        "Is Bijan Robinson a buy or sell right now?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_query"] = ex

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about a player, trade, or rookie...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Running ML models and retrieving scouting context..."):
            try:
                answer = ask_dynasty_scout(query)
            except Exception as e:
                answer = f"Something went wrong: {e}"
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
