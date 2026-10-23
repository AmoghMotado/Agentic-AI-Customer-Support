"""The agent: graph state, LLM setup and one function per graph node.

Each node receives the current state and returns ONLY the fields it changes.
LangGraph merges those changes into the state.
"""

import json
import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src import config, db, prompts
from src.rag import KnowledgeBaseError, retrieve
from src.tools import LLM_TOOLS, execute_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SupportState(TypedDict):
    # Conversation so far (user + assistant text). add_messages appends new ones.
    messages: Annotated[list[AnyMessage], add_messages]
    # Which business this conversation belongs to. Set by the app, never by the LLM.
    # Retrieval and every tool only use this business's data.
    business_id: str
    # Business name and one-line description, used to fill in the prompts
    business_name: str
    business_description: str
    # Customer ID from the sidebar ("" if not given). Tools that need it use this.
    customer_id: str
    # Router decision: "rag" | "tool" | "direct" | "clarify"
    action: str
    # Short human-readable label of the request, shown in debug mode
    route_label: str
    # For RAG: the standalone question used for vector search
    search_query: str
    # For clarify: what information is missing
    missing_info: str
    # For RAG: chunks returned by FAISS (source, chunk, score, content)
    retrieved_context: list[dict]
    # For tools: the LLM's tool requests and the tool results in this turn only.
    # Kept separate from `messages` so chat history stays clean.
    tool_messages: list[AnyMessage]
    # For debug mode: [{"tool", "input", "result"}, ...]
    tool_calls: list[dict]
    # How many tool rounds have run this turn (safety limit)
    tool_rounds: int
    # The answer shown to the user
    final_response: str


class RouteDecision(BaseModel):
    """Structured output the router LLM must return."""

    action: Literal["rag", "tool", "direct", "clarify"] = Field(
        description="How to handle the latest customer message."
    )
    label: str = Field(description="2 to 5 word label for the request.")
    search_query: str = Field(
        description="Standalone question for knowledge-base search if action is rag, else empty."
    )
    missing_info: str = Field(
        description="What information is missing if action is clarify, else empty."
    )


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def api_key_for(business_id: str) -> str:
    """The business's own OpenAI key if it has one, otherwise the key from .env."""
    return db.get_business_api_key(business_id) or config.OPENAI_API_KEY


def get_llm(state: SupportState) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        api_key=api_key_for(state["business_id"]),
        timeout=60,
        max_retries=2,
    )


def _prompt(template: str, state: SupportState, **extra) -> str:
    """Fill a prompt template with the business's name and description."""
    return template.format(
        business_name=state["business_name"],
        business_description=state.get("business_description") or "an online business",
        **extra,
    )


def _customer_label(state: SupportState) -> str:
    return state.get("customer_id") or "not provided"


def _history(state: SupportState) -> list[AnyMessage]:
    """The most recent chat messages, so the prompt doesn't grow forever."""
    return state["messages"][-config.MAX_HISTORY_MESSAGES:]


def _last_user_text(state: SupportState) -> str:
    return state["messages"][-1].content if state["messages"] else ""


def _finish(text: str) -> dict:
    """State update that delivers the final answer and records it in history."""
    text = (text or "").strip() or "Sorry, I couldn't generate a response. Please try again."
    return {"final_response": text, "messages": [AIMessage(content=text)]}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def router_node(state: SupportState) -> dict:
    """Decide which branch handles the message: rag, tool, direct or clarify."""
    router = get_llm(state).with_structured_output(RouteDecision)
    system = _prompt(prompts.ROUTER_PROMPT, state, customer_id=_customer_label(state))
    decision: RouteDecision = router.invoke([SystemMessage(content=system)] + _history(state))
    logger.info("Route: %s (%s)", decision.action, decision.label)
    return {
        "action": decision.action,
        "route_label": decision.label,
        "search_query": decision.search_query.strip() or _last_user_text(state),
        "missing_info": decision.missing_info.strip(),
    }


def retrieve_node(state: SupportState) -> dict:
    """RAG step 1: search this business's FAISS index for related chunks."""
    try:
        chunks = retrieve(state["business_id"], state["search_query"])
    except KnowledgeBaseError as exc:
        logger.error("Retrieval failed: %s", exc)
        chunks = []
    return {"retrieved_context": chunks}


def generate_node(state: SupportState) -> dict:
    """RAG step 2: answer the question using only the retrieved chunks."""
    chunks = state.get("retrieved_context", [])
    if not chunks:
        return _finish(
            f"Sorry, I can't access {state['business_name']}'s policy information right now. "
            "Please try again in a moment, or I can create a support ticket for you."
        )

    context = "\n\n".join(f"[{c['source']}]\n{c['content']}" for c in chunks)
    system = _prompt(prompts.RAG_ANSWER_PROMPT, state, context=context)
    answer = get_llm(state).invoke([SystemMessage(content=system)] + _history(state))
    return _finish(answer.text)


def tool_agent_node(state: SupportState) -> dict:
    """Tool branch: the LLM picks a tool (or, after results, writes the answer)."""
    system = _prompt(prompts.TOOL_AGENT_PROMPT, state, customer_id=_customer_label(state))
    messages = [SystemMessage(content=system)] + _history(state) + state["tool_messages"]

    llm = get_llm(state)
    if state["tool_rounds"] < config.MAX_TOOL_ROUNDS:
        llm = llm.bind_tools(LLM_TOOLS)
    # After MAX_TOOL_ROUNDS the LLM gets no tools, so it must write an answer.

    response = llm.invoke(messages)
    if response.tool_calls:
        # Save the request; the graph routes to run_tools next.
        return {"tool_messages": state["tool_messages"] + [response]}
    return _finish(response.text)


def run_tools_node(state: SupportState) -> dict:
    """Execute the tools the LLM requested, with validation and error handling."""
    last = state["tool_messages"][-1]
    new_tool_messages = []
    log = list(state.get("tool_calls", []))

    for call in last.tool_calls:
        result = execute_tool(
            call["name"], call.get("args", {}), state["business_id"], state.get("customer_id", "")
        )
        logger.info("Tool %s(%s) -> ok=%s", call["name"], call.get("args"), result.get("ok"))
        log.append({"tool": call["name"], "input": call.get("args", {}), "result": result})
        new_tool_messages.append(
            ToolMessage(content=json.dumps(result), tool_call_id=call["id"], name=call["name"])
        )

    return {
        "tool_messages": state["tool_messages"] + new_tool_messages,
        "tool_calls": log,
        "tool_rounds": state["tool_rounds"] + 1,
    }


def direct_node(state: SupportState) -> dict:
    """Plain conversational reply (greetings, thanks, small talk)."""
    system = _prompt(prompts.DIRECT_PROMPT, state)
    reply = get_llm(state).invoke([SystemMessage(content=system)] + _history(state))
    return _finish(reply.text)


def clarify_node(state: SupportState) -> dict:
    """Ask the customer for the missing information instead of guessing."""
    missing = state.get("missing_info") or "a few more details about your request"
    system = _prompt(prompts.CLARIFY_PROMPT, state, missing_info=missing)
    reply = get_llm(state).invoke([SystemMessage(content=system)] + _history(state))
    return _finish(reply.text)


# ---------------------------------------------------------------------------
# Conditional edges (functions that pick the next node)
# ---------------------------------------------------------------------------
def route_by_action(state: SupportState) -> str:
    return state["action"]


def after_tool_agent(state: SupportState) -> str:
    """Go to run_tools if the LLM just asked for a tool, otherwise finish."""
    if state.get("final_response"):
        return "end"
    return "run_tools"
