"""LangGraph workflow: wires the agent nodes together.

    START -> router -> rag     -> retrieve -> generate -> END
                    -> tool    -> tool_agent <-> run_tools
                                  tool_agent -> END
                    -> direct  -> direct -> END
                    -> clarify -> clarify -> END

Print the graph as a Mermaid diagram with:
    python -m src.graph
"""

import logging

import openai
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src import config, db
from src.agent import (
    api_key_for,
    SupportState,
    after_tool_agent,
    clarify_node,
    direct_node,
    generate_node,
    retrieve_node,
    route_by_action,
    router_node,
    run_tools_node,
    tool_agent_node,
)
from src.rag import KnowledgeBaseError

logger = logging.getLogger(__name__)


def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("run_tools", run_tools_node)
    graph.add_node("direct", direct_node)
    graph.add_node("clarify", clarify_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_by_action,
        {"rag": "retrieve", "tool": "tool_agent", "direct": "direct", "clarify": "clarify"},
    )

    # RAG branch
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Tool branch: loop until the LLM stops asking for tools
    graph.add_conditional_edges(
        "tool_agent", after_tool_agent, {"run_tools": "run_tools", "end": END}
    )
    graph.add_edge("run_tools", "tool_agent")

    graph.add_edge("direct", END)
    graph.add_edge("clarify", END)

    return graph.compile()


# Compiled once and reused for every message.
support_graph = build_graph()


def _to_langchain_messages(history: list[dict]) -> list:
    """Convert Streamlit-style history [{"role", "content"}] to LangChain messages."""
    messages = []
    for item in history[-config.MAX_HISTORY_MESSAGES:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages


def _friendly_error(exc: Exception) -> str:
    """Turn technical exceptions into messages a normal user can act on."""
    if isinstance(exc, openai.AuthenticationError):
        return ("The OpenAI API key is invalid. Check OPENAI_API_KEY in your .env file, "
                "or this business's own key on the Admin page.")
    if isinstance(exc, openai.RateLimitError):
        return ("The OpenAI API rate limit or quota was reached. Check your billing/credits "
                "on platform.openai.com, then try again.")
    if isinstance(exc, openai.NotFoundError):
        return (f"The model '{config.OPENAI_MODEL}' is not available for your API key. "
                "Set OPENAI_MODEL in .env to a model your account can use.")
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "I couldn't reach the OpenAI API. Please check your internet connection and try again."
    if isinstance(exc, openai.APIError):
        return "The AI service returned an error. Please try again in a moment."
    if isinstance(exc, KnowledgeBaseError):
        return f"The knowledge base is unavailable: {exc}"
    return "Sorry, something went wrong while handling your request. Please try again."


def run_turn(user_message: str, history: list[dict], business_id: str, customer_id: str = "") -> dict:
    """Run one user message through the graph, for one business.

    Returns a dict that the UI can display directly:
        {"response", "action", "route_label", "search_query", "sources",
         "tool_calls", "error"}
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return {"response": "Please type a message so I can help you.", "action": "none",
                "error": "empty message"}

    business = db.get_business(business_id)
    if business is None:
        return {"response": "This support assistant is not linked to a valid business.",
                "action": "none", "error": "unknown business"}

    if not config.key_looks_real(api_key_for(business_id)):
        return {"response": "OPENAI_API_KEY is not set. Add it to your .env file and restart the app.",
                "action": "none", "error": "missing API key"}

    initial_state = {
        "messages": _to_langchain_messages(history) + [HumanMessage(content=user_message)],
        "business_id": business_id,
        "business_name": business["name"],
        "business_description": business["description"],
        "customer_id": (customer_id or "").strip().upper(),
        "action": "",
        "route_label": "",
        "search_query": "",
        "missing_info": "",
        "retrieved_context": [],
        "tool_messages": [],
        "tool_calls": [],
        "tool_rounds": 0,
        "final_response": "",
    }

    try:
        final = support_graph.invoke(initial_state)
    except Exception as exc:  # shown nicely to the user, full details in the terminal
        logger.exception("Graph run failed")
        return {"response": _friendly_error(exc), "action": "error", "error": type(exc).__name__}

    return {
        "response": final["final_response"],
        "action": final["action"],
        "route_label": final.get("route_label", ""),
        "search_query": final.get("search_query", "") if final["action"] == "rag" else "",
        "missing_info": final.get("missing_info", ""),
        "sources": [
            {"source": c["source"], "chunk": c["chunk"], "score": c["score"]}
            for c in final.get("retrieved_context", [])
        ],
        "tool_calls": final.get("tool_calls", []),
        "business_id": business_id,
        "error": "",
    }


if __name__ == "__main__":
    print(support_graph.get_graph().draw_mermaid())
