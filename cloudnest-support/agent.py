"""
LangGraph ReAct agent for CloudNest support.

Two tools:
  - search_knowledge_base: wraps the RAG retriever
  - create_support_ticket: mocked, returns a fake ticket id

Tools are defined at module level (not inside a closure) so the @tool
decorator can reliably read their docstrings across LangChain versions.
The retriever is injected via a module-level variable that build_agent sets.
"""

import random
import string

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


SYSTEM_PROMPT = """You are CloudNest Support, the official virtual assistant for 
CloudNest — a cloud file-storage platform. You were built by Eng. Youssef Bastawisy.

# Role
Help customers resolve questions about their CloudNest account, subscription, and 
product usage. Be helpful, accurate, and professional at all times. Never speculate 
or fabricate policy details.

# Scope of the knowledge base
The knowledge base covers: account management and password reset, pricing and plans, 
file sharing, refunds and cancellation, and troubleshooting. Treat anything outside 
these areas as out of scope.

# Tool usage
1. `search_knowledge_base` — For any question about CloudNest's product, pricing, 
billing, policies, or troubleshooting, call this tool before answering. Do not 
rely on prior knowledge.
   - Results are returned as one or more chunks in the format 
`[Source N: filename]\\n<content>`. Treat this content as authoritative. If a 
returned chunk contains the answer (in full or in part), use it directly; do not 
state that the knowledge base is empty.
   - Call this tool at most twice per user turn. If the first call returns 
relevant content, answer from it. Only retry once, with a reworded query, when 
the first result is clearly unrelated.
   - If two searches yield no relevant information, acknowledge this honestly and 
offer to open a support ticket.

2. `create_support_ticket` — Call this when the user explicitly asks to speak with 
a human, files a complaint, or reports an issue the knowledge base cannot resolve. 
Provide a concise issue summary and an appropriate priority.

3. No tool — For greetings, thanks, or off-topic small talk, reply directly and 
briefly without invoking any tool.

# Response style
- Be concise: typically 2–4 sentences. Use short bullet lists only when the answer 
is naturally enumerable (e.g. plan tiers).
- Cite the source filename inline whenever you used `search_knowledge_base` 
(e.g. "See `02_pricing_and_plans.md`.").
- Maintain a professional, courteous tone. Avoid filler, marketing language, and 
emojis unless the user uses them first.
- If a user request cannot be completed, explain why and propose a clear next step 
(typically: opening a ticket)."""


# Module-level retriever holder. build_agent() sets this before tools run.
_RETRIEVER = None


@tool
def search_knowledge_base(query: str) -> str:
    """Search CloudNest's help center for information about pricing, features,
    troubleshooting, account management, sharing, refunds, and policies.
    Use this for any question about how CloudNest works.

    Args:
        query: A natural-language search query.
    """
    if _RETRIEVER is None:
        return "Retriever not initialized."
    docs = _RETRIEVER.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."

    formatted = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown").replace("\\", "/").split("/")[-1]
        formatted.append(f"[Source {i}: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


@tool
def create_support_ticket(issue_summary: str, priority: str = "normal") -> str:
    """Create a support ticket for issues that cannot be resolved from the
    knowledge base, or when the user explicitly asks to talk to a human.

    Args:
        issue_summary: A 1-2 sentence summary of the user's issue.
        priority: One of "low", "normal", "high", "urgent". Default "normal".
    """
    ticket_id = "CN-" + "".join(random.choices(string.digits, k=6))
    return (
        f"Ticket created successfully.\n"
        f"  Ticket ID: {ticket_id}\n"
        f"  Priority: {priority}\n"
        f"  Summary: {issue_summary}\n"
        f"A support engineer will email you within 24 hours."
    )


def build_agent(retriever, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.2):
    """Build the LangGraph ReAct agent.

    Args:
        retriever: a LangChain retriever (from rag.build_default_vectorstore().as_retriever())
        model_name: Groq model id
        temperature: 0.0-1.0; lower = more deterministic
    """
    global _RETRIEVER
    _RETRIEVER = retriever

    llm = ChatGroq(model=model_name, temperature=temperature)
    tools = [search_knowledge_base, create_support_ticket]
    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent


if __name__ == "__main__":
    import os
    from rag import build_default_vectorstore
    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY env var to run this test.")
        raise SystemExit(1)

    print("Building vectorstore...")
    vs = build_default_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    print("Building agent...")
    agent = build_agent(retriever)

    config = {"configurable": {"thread_id": "test-1"}}
    questions = [
        "Hi!",
        "What's the difference between the Pro and Business plans?",
        "I want to talk to a human, my files are corrupted and I can't access them.",
    ]

    for q in questions:
        print(f"\n👤 {q}")
        result = agent.invoke({"messages": [("user", q)]}, config=config)
        print(f"🤖 {result['messages'][-1].content}")