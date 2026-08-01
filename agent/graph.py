import logging
import os

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

from agent.state import DynastyState, ML_TOOLS, RAG_TOOLS

logger = logging.getLogger(__name__)

# Model used for routing/classification — fast and cheap, this is a simple task
SUPERVISOR_MODEL = "claude-sonnet-4-6"

# Model used for sub-agent reasoning and final report writing — needs to
# synthesize numeric ML output with qualitative scouting text coherently
WORKER_MODEL = "claude-sonnet-4-6"


def _get_llm(model: str = WORKER_MODEL, temperature: float = 0.3):
    """
    Centralized LLM factory. Temperature is kept low (0.3) because this agent
    reports on statistical projections — we want grounded, consistent prose,
    not creative variation between runs on the same input.
    """
    return ChatAnthropic(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Node: Supervisor
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """You are the routing supervisor for a dynasty fantasy football AI scout.

Classify the user's query into exactly one category:
  - "nfl_analysis"     — questions about established NFL players (performance, injury risk, projections)
  - "college_scouting" — questions about incoming rookies / draft prospects with no NFL history
  - "dynasty_advice"    — roster construction, trade evaluation, buy/sell/hold, "who should I draft"
                          questions that need BOTH player projections and broader context

Also extract:
  - player_name: the primary player being discussed (empty string if none, or if multiple — pick the main one)
  - season: the NFL season year being asked about (use the upcoming season if ambiguous; default 2025)

Respond with EXACTLY this format, nothing else:
CATEGORY: <category>
PLAYER: <player_name>
SEASON: <season>
"""


def supervisor_node(state: DynastyState) -> dict:
    """
    Classify the incoming query and extract routing parameters.
    Uses structured text output (not JSON mode) parsed with simple string
    splitting — reliable enough for this 3-field extraction and avoids
    JSON-parsing brittleness with Claude's text completions.
    """
    llm = _get_llm(model=SUPERVISOR_MODEL, temperature=0.0)
    user_query = state["messages"][-1].content if state["messages"] else ""

    response = llm.invoke([
        {"role": "system", "content": SUPERVISOR_PROMPT},
        {"role": "user", "content": user_query},
    ])

    parsed = _parse_supervisor_output(response.content)
    logger.info(f"Supervisor routed to: {parsed}")

    return {
        "query_type": parsed["category"],
        "player_name": parsed["player_name"],
        "season": parsed["season"],
    }


def _parse_supervisor_output(text: str) -> dict:
    """Parse the supervisor's structured text response into routing fields."""
    result = {"category": "dynasty_advice", "player_name": "", "season": 2025}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            cat = line.split(":", 1)[1].strip().lower()
            if cat in ("nfl_analysis", "college_scouting", "dynasty_advice"):
                result["category"] = cat
        elif line.upper().startswith("PLAYER:"):
            result["player_name"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SEASON:"):
            try:
                result["season"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return result


def route_query(state: DynastyState) -> str:
    """Conditional edge function — maps query_type to the next node name."""
    return {
        "nfl_analysis": "nfl_analyst",
        "college_scouting": "college_scout",
        "dynasty_advice": "dynasty_advisor",
    }.get(state["query_type"], "dynasty_advisor")


# ---------------------------------------------------------------------------
# Sub-agent hand-off helpers
# ---------------------------------------------------------------------------

def _extract_tool_evidence(messages: list) -> str:
    """
    Pull the raw tool outputs out of a react-agent message trace and render them
    as plain text.

    The report writer is instructed to ground itself ONLY in tool outputs, so it
    needs to actually SEE them. We render them as text (rather than forwarding the
    structured tool_use/tool_result message pairs) because the downstream
    report-writer LLM is not bound to any tools — replaying raw tool_result blocks
    to a tool-less model is invalid for the Anthropic API.
    """
    from langchain_core.messages import ToolMessage

    blocks = []
    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", None) or "tool"
            content = m.content if isinstance(m.content, str) else str(m.content)
            blocks.append(f"[{name}]\n{content.strip()}")
    return "\n\n".join(blocks)


def _finalize_subagent(result: dict) -> dict:
    """
    Collapse a react-agent result into a single AIMessage for the report writer.

    Combines the sub-agent's synthesized answer with the raw tool evidence so the
    report writer has verifiable numbers to work from — otherwise it only sees the
    analyst's prose and may (correctly, per its prompt) refuse to trust unattributed
    statistics, producing a "this looks fabricated" non-answer.
    """
    from langchain_core.messages import AIMessage

    messages = result.get("messages", [])
    final = messages[-1].content if messages else ""
    if isinstance(final, list):  # Anthropic can return content blocks
        final = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in final)

    evidence = _extract_tool_evidence(messages)
    content = final
    if evidence:
        content = f"{final}\n\n---\nTOOL OUTPUTS (raw evidence the analysis is based on):\n\n{evidence}"

    return {"messages": [AIMessage(content=content)]}


# ---------------------------------------------------------------------------
# Node: NFL Analyst — established veterans
# ---------------------------------------------------------------------------

NFL_ANALYST_PROMPT = """You are an NFL fantasy football analyst with access to:
  - predict_nfl_performance: ML-based next-season PPG projection with floor/ceiling
  - predict_injury_risk: calibrated injury probability for next season
  - retrieve_player_context: scouting reports, bio, and game log context
  - retrieve_injury_context: detailed injury history
  - retrieve_news_context: recent transactions, depth chart, and news

For player analysis questions, ALWAYS call predict_nfl_performance first to
ground your answer in the model's projection, then retrieve_news_context to
check for any very recent situational changes (injuries, trades, depth chart
moves) the static model wouldn't know about. Use retrieve_injury_context when
injury history specifically matters to the question.

Be concise. Report concrete numbers from the tools, not vague impressions.
"""


def build_nfl_analyst():
    llm = _get_llm()
    return create_react_agent(llm, ML_TOOLS + RAG_TOOLS, prompt=NFL_ANALYST_PROMPT)


def nfl_analyst_node(state: DynastyState) -> dict:
    agent = build_nfl_analyst()
    query = f"Player: {state['player_name']}, Season: {state['season']}. {state['messages'][-1].content}"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _finalize_subagent(result)


# ---------------------------------------------------------------------------
# Node: College Scout — incoming prospects
# ---------------------------------------------------------------------------

COLLEGE_SCOUT_PROMPT = """You are a college football scout specializing in NFL
dynasty rookie evaluation. You have access to:
  - predict_college_prospect: ML projection (Ridge + historical comps) for
    year-3 NFL fantasy PPG based on college production and combine athleticism
  - retrieve_college_scouting: draft profile text, strengths/weaknesses,
    college production narrative
  - retrieve_player_context: any available bio/scouting context
  - on_demand_ingest: use this if retrieval returns no results for a prospect,
    then retry retrieval

ALWAYS call predict_college_prospect first for a quantitative anchor, then
retrieve_college_scouting for qualitative context (playing style, scheme fit,
draft analyst commentary) to round out the projection.

If a prospect has no NFL feature data, clearly state this is a college-stats-only
projection and explain the model's archetype classification (e.g. "ELITE_PROSPECT",
"PRODUCTION_BASED") in plain language.
"""


def build_college_scout():
    llm = _get_llm()
    return create_react_agent(llm, ML_TOOLS + RAG_TOOLS, prompt=COLLEGE_SCOUT_PROMPT)


def college_scout_node(state: DynastyState) -> dict:
    agent = build_college_scout()
    query = f"Prospect: {state['player_name']}, Draft season: {state['season']}. {state['messages'][-1].content}"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _finalize_subagent(result)


# ---------------------------------------------------------------------------
# Node: Dynasty Advisor — roster strategy, trades, buy/sell
# ---------------------------------------------------------------------------

DYNASTY_ADVISOR_PROMPT = """You are a dynasty fantasy football strategy advisor.
You have access to the full tool suite:
  - predict_nfl_performance, predict_injury_risk, predict_college_prospect
  - retrieve_player_context, retrieve_injury_context, retrieve_news_context,
    retrieve_college_scouting

Dynasty questions often require comparing MULTIPLE players or weighing
short-term production against long-term asset value (age curve, contract
situation, draft capital invested). Call the relevant prediction tool for
EACH player mentioned in the question before forming a recommendation.

Always ground buy/sell/hold and trade recommendations in the actual model
outputs (PPG projections, injury risk, confidence level) — never give a
recommendation without first calling the relevant tools for every player involved.
"""


def build_dynasty_advisor():
    llm = _get_llm()
    return create_react_agent(llm, ML_TOOLS + RAG_TOOLS, prompt=DYNASTY_ADVISOR_PROMPT)


def dynasty_advisor_node(state: DynastyState) -> dict:
    agent = build_dynasty_advisor()
    query = state["messages"][-1].content
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _finalize_subagent(result)


# ---------------------------------------------------------------------------
# Node: Report Writer — final synthesis pass, no tools
# ---------------------------------------------------------------------------

REPORT_WRITER_PROMPT = """You are finalizing a dynasty fantasy football scouting
report. The preceding assistant message is the work of a specialist analyst
agent that already ran the ML and retrieval tools; it contains that analyst's
conclusion followed by a "TOOL OUTPUTS (raw evidence...)" section with the
verbatim outputs of those tools (PPG projections, injury risk, confidence
levels, comps, and retrieved scouting context).

Treat that analyst message — including its TOOL OUTPUTS section — as the
authoritative, already-verified source data. These numbers came from real tools;
do NOT dismiss them as unverified or fabricated.

Write a clear, well-organized final answer that:
  1. Leads with the concrete numbers from the tool outputs (don't bury the projection)
  2. Incorporates qualitative scouting context to explain WHY the model says what it says
  3. Notes confidence/uncertainty honestly — if the model's confidence was LOW
     or MEDIUM, say so; if a tool reported no data (e.g. injury risk unavailable),
     state that plainly rather than inventing a value
  4. Stays grounded ONLY in information present in the analyst message and its
     tool outputs above — do not invent statistics, comps, or news that weren't retrieved

Do not call any tools. Just synthesize what's already in the conversation.
"""


def report_writer_node(state: DynastyState) -> dict:
    llm = _get_llm(temperature=0.4)  # slightly higher — this is prose synthesis, not tool routing

    # The sub-agent's answer is the last message and is an AIMessage. If we ended
    # the prompt on that assistant turn, Claude would treat it as a prefill and
    # continue it rather than writing a fresh report. Append an explicit user turn
    # so the model produces a clean, self-contained synthesis.
    response = llm.invoke([
        {"role": "system", "content": REPORT_WRITER_PROMPT},
        *state["messages"],
        {"role": "user", "content": (
            "Using only the analysis and tool outputs above, write the final "
            "dynasty scouting report now."
        )},
    ])

    return {
        "final_report": response.content,
        "messages": [response],
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_dynasty_graph():
    """
    Compile the full supervisor → sub-agent → report-writer graph.

    Usage:
        graph = build_dynasty_graph()
        result = graph.invoke({
            "messages": [{"role": "user", "content": "Should I trade for Justin Jefferson?"}]
        })
        print(result["final_report"])
    """
    builder = StateGraph(DynastyState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("nfl_analyst", nfl_analyst_node)
    builder.add_node("college_scout", college_scout_node)
    builder.add_node("dynasty_advisor", dynasty_advisor_node)
    builder.add_node("report_writer", report_writer_node)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "nfl_analyst": "nfl_analyst",
            "college_scout": "college_scout",
            "dynasty_advisor": "dynasty_advisor",
        },
    )

    builder.add_edge("nfl_analyst", "report_writer")
    builder.add_edge("college_scout", "report_writer")
    builder.add_edge("dynasty_advisor", "report_writer")
    builder.add_edge("report_writer", END)

    return builder.compile()


# Module-level compiled graph — import this directly for simple use cases
dynasty_graph = None


def get_compiled_graph():
    """Lazily compile and cache the graph (compilation has some overhead)."""
    global dynasty_graph
    if dynasty_graph is None:
        dynasty_graph = build_dynasty_graph()
    return dynasty_graph
