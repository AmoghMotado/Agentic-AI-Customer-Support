# Interview Preparation

## 30-second explanation

I built an AI customer support assistant for a fictional online store. For each message, a LangGraph agent decides whether to answer from the company's policy documents using RAG, call a tool like order-status lookup or ticket creation, reply conversationally, or ask for missing details such as an order ID. It uses OpenAI for the LLM, FAISS for vector search and Streamlit for the chat UI.

## 1-minute explanation

The project is an agentic customer support assistant for a demo e-commerce store called NovaCart. The key problem is routing: some questions need policy knowledge, some need account data or an action, and some can't be handled yet because information is missing.

I built a LangGraph workflow where a router node uses the LLM with structured output to classify each message into RAG, tool, direct or clarify. Policy questions go through a RAG pipeline: documents are split into chunks, embedded with a sentence-transformers model and stored in FAISS, and the LLM answers using only the retrieved chunks. Action requests go to a tool-calling node where the LLM picks a function like `get_order_status` or `create_support_ticket`, and my code validates the arguments before running it. The UI is a Streamlit chat with a debug mode that shows which route and tools were used. The order and customer systems are simulated with JSON files.

## 2-minute technical explanation

There are four layers.

1. **UI**: Streamlit keeps the chat history in session state and calls one function, `run_turn`, for each message, passing the history and the customer ID from the sidebar.
2. **Orchestration**: `run_turn` invokes a compiled LangGraph `StateGraph`. The state is a typed dictionary with the messages, customer ID, the router's decision, retrieved chunks, a tool log and the final response.
3. **Agent nodes**: the router calls the LLM with `with_structured_output` and a Pydantic schema, so it returns a validated object like `{action: "tool", label: "order status"}` instead of free text. A conditional edge maps that action to a branch.
   - RAG branch: `retrieve` runs a FAISS similarity search on a standalone version of the question (the router rewrites follow-ups), and `generate` prompts the LLM to answer only from those chunks and to admit when the answer isn't there.
   - Tool branch: `tool_agent` calls the LLM with tools bound via `bind_tools`. If it returns tool calls, `run_tools` executes them and the graph loops back so the LLM can write the final answer from the results. A round limit prevents infinite loops.
   - Direct and clarify branches are single LLM calls with their own prompts.
4. **Tools and data**: plain Python functions over JSON files. They validate ID formats, check that the order belongs to the customer, and return `ok: false` errors instead of raising exceptions.

Errors from OpenAI (bad key, quota, network) are caught in `run_turn` and converted into readable messages.

## Architecture explanation

Streamlit UI → `run_turn()` → LangGraph graph (router → one of four branches) → final response back to the UI. The RAG branch uses the FAISS index built from `data/knowledge_base`. The tool branch uses `src/tools.py` over `data/*.json`. All prompts live in `src/prompts.py` and all settings in `src/config.py`, loaded from `.env`.

## RAG explanation

RAG (retrieval-augmented generation) gives the LLM relevant facts at question time instead of relying on what it memorised in training. Offline, I load 8 policy documents, split them into ~500-character overlapping chunks, convert each chunk into a 384-dimensional embedding vector and store them in FAISS. At question time, the question is embedded with the same model, FAISS returns the 4 nearest chunks, and those chunks are put in the prompt with an instruction to answer only from them. This keeps answers grounded in NovaCart's actual policies and lets me show the sources.

## Agent explanation

Here, an agent means an LLM that decides what to do next instead of following a fixed script. My agent decides between retrieval, calling tools, replying directly or asking a question, and in the tool branch it decides which function to call with which arguments. The decisions are limited to a small, explicit set of options, which makes the behaviour predictable and testable.

## LangGraph explanation

LangGraph models a workflow as a graph of nodes (Python functions) connected by edges, sharing a state object. Each node returns only the fields it changes. Conditional edges choose the next node from the state, which is how the router branches and how the tool loop works (`tool_agent` → `run_tools` → `tool_agent` until no more tool calls). I can print the graph as a Mermaid diagram with `python -m src.graph`.

## Tool calling explanation

I describe each tool to the LLM with a name, a docstring and typed arguments (LangChain's `@tool` turns that into a JSON schema). When the LLM decides a tool is needed, it doesn't run anything: it returns a structured tool call such as `get_order_status(order_id="ORD1001")`. My code validates and executes it, returns the result as a `ToolMessage`, and the LLM turns that result into a natural answer.

## Why FAISS?

It is a fast, free, local vector search library. For a small knowledge base it needs no server or account, and the index saves to a folder. A hosted vector database would add setup and cost without benefit at this size.

## Why LangChain?

I use it only where it saves code: document loading, text splitting, the embeddings and FAISS wrappers, the OpenAI chat model with `bind_tools` and `with_structured_output`, and message types that LangGraph understands. I did not use its older high-level agent classes.

## Why LangGraph?

It makes the agent's control flow explicit. I can see and explain every possible path, add a loop limit, and keep shared state in one typed object. A single "agent executor" hides that flow, which makes it harder to debug and to put safeguards in specific places.

## Why Streamlit?

It builds a chat UI in pure Python with `st.chat_message` and `st.chat_input`, so I could focus on the AI part instead of frontend code.

## How does the system decide between RAG and tools?

A router node sends the conversation and a routing prompt to the LLM and forces a structured answer with one of four actions. The prompt has explicit rules: policy questions → RAG; requests needing a specific order or account → tool; missing order or customer ID → clarify; greetings → direct. A conditional edge in LangGraph then sends the state to the matching branch. Debug mode shows the chosen action for every message.

## What happens if a tool fails?

Tools never raise exceptions to the user. Validation problems and missing records return `{"ok": false, "error": "..."}`, and unexpected exceptions are caught in `execute_tool` and logged to the terminal. The error goes back to the LLM as the tool result, and the prompt tells it to explain the problem politely and suggest a next step.

## How do you prevent incorrect tool execution?

Several layers:
1. The router sends requests with missing IDs to clarification instead of the tool branch.
2. Prompts and tool descriptions say never to invent IDs.
3. Code validates ID formats and checks that records exist.
4. The signed-in customer ID from the app overrides any customer ID the LLM sends, and customers can only view or cancel their own orders.
5. Unknown tools and unexpected arguments are rejected.
6. `cancel_order` only works for orders still in Processing.
7. A maximum of 3 tool rounds per message.

## How could this be improved in production?

Real authentication and a database instead of JSON; human confirmation before irreversible actions (LangGraph interrupts); an evaluation set to measure routing accuracy and answer faithfulness; monitoring and tracing (for example LangSmith); persistent conversation memory; streaming responses; rate limiting; and hybrid search with reranking for a larger knowledge base.

## Interview questions and answers

1. **What is an embedding?** A list of numbers representing the meaning of a text. Texts with similar meaning have vectors that are close together, which is what makes semantic search possible.

2. **Why split documents into chunks?** Whole documents dilute the meaning of the embedding and waste prompt space. Small chunks let search return just the relevant paragraph. Overlap keeps sentences that cross a boundary from losing context.

3. **How did you choose chunk size and top-k?** The policy documents have short sections, so ~500 characters keeps one topic per chunk, and 4 chunks give enough context without flooding the prompt. I checked with test questions that the right file is retrieved.

4. **What similarity measure does FAISS use here?** L2 (Euclidean) distance on normalised vectors, which ranks results the same way as cosine similarity. Lower distance means more similar.

5. **How do you stop the LLM from hallucinating policies?** The RAG prompt says to answer only from the retrieved excerpts and to say when the answer isn't there. I test this with a question about a policy that doesn't exist, like price matching.

6. **What is the difference between RAG and fine-tuning?** RAG adds knowledge at query time by retrieving documents, so updating a policy means editing a text file. Fine-tuning changes model weights and is better for style or behaviour than for facts that change.

7. **What is structured output and why use it for routing?** The LLM is forced to return JSON matching a schema (my Pydantic `RouteDecision`). That means the route is always one of four valid values and the code never has to parse free text.

8. **What is in your LangGraph state and why?** Messages for context, the customer ID for tools, the route for branching, retrieved chunks for the answer and sources, a tool log for debug mode, a round counter for safety, and the final response.

9. **Why keep tool messages separate from chat history?** Tool requests and results are only needed within one turn. Keeping them out of the history keeps prompts small and avoids sending broken tool-call/result pairs when old history is trimmed.

10. **How does the conversation handle follow-ups?** Streamlit passes recent history into the graph. The router sees it, so "ORD1001" after "Where is my order?" continues the order lookup, and it rewrites follow-up policy questions into standalone search queries.

11. **Why not let the LLM call the RAG search as just another tool?** That works too. I chose an explicit router because it makes the decision visible, easier to test, and maps directly to the four behaviours the assistant must have. The trade-off is that a message mixing a question and an action is handled by one branch.

12. **What does `bind_tools` do?** It attaches the tools' JSON schemas to every request to the model, so the model can respond with tool calls instead of text.

13. **Why use a local embedding model?** It's free, works offline after the first download, and is good enough for a small English knowledge base. I can switch to OpenAI embeddings with one setting in `.env`.

14. **How do you handle a missing or invalid API key?** The app checks for the key on start and shows a clear message with the input disabled. If the key is invalid, the OpenAI authentication error is caught and shown as a readable message instead of a stack trace.

15. **How did you test it?** Pytest tests for every tool (valid, invalid, missing and unauthorised inputs), document loading, chunking and retrieval accuracy, plus a manual checklist for routing, multi-turn flows and failure cases.

16. **What would you change for many users?** Replace JSON files with a database, add real login, cache the embedding model in a separate service, and add tracing to monitor routing mistakes and cost.

17. **What are the limitations?** Simulated data, one route per message, no authentication, history lost on refresh, and routing that depends on the LLM following the prompt.
