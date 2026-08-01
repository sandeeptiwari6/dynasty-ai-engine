import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ask_dynasty_scout(query: str) -> str:
    """
    Run a single query through the full agent graph and return the final report.
    This is the one function the Streamlit app and any other caller needs.
    """
    from agent.graph import get_compiled_graph

    graph = get_compiled_graph()
    result = graph.invoke({
        "messages": [{"role": "user", "content": query}],
        "query_type": "",
        "player_name": "",
        "season": 2025,
        "ml_prediction": {},
        "rag_context": "",
        "final_report": "",
    })
    return result.get("final_report", "No report generated.")


def run_interactive():
    """Simple REPL for manual testing without the Streamlit UI."""
    print("=" * 60)
    print("Dynasty Scout AI — interactive mode")
    print("Type a question, or 'quit' to exit.")
    print("=" * 60)

    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        try:
            answer = ask_dynasty_scout(query)
            print(f"\n{answer}\n")
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynasty-scout agent runner")
    parser.add_argument("--query", type=str, help="Single query to run")
    parser.add_argument("--interactive", action="store_true", help="Run interactive REPL")
    args = parser.parse_args()

    if args.query:
        print(ask_dynasty_scout(args.query))
    elif args.interactive:
        run_interactive()
    else:
        parser.print_help()
