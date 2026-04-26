import os
import streamlit as st

from rag import build_default_vectorstore
from agent import build_agent


# ---------- Defaults (formerly user-tweakable in the sidebar) ----------
# Note: llama-3.3-70b-versatile occasionally hits Groq's `tool_use_failed`
# error because it emits Meta's legacy <function=...> XML syntax instead of
# the structured tool_calls JSON. 8b-instant follows the protocol reliably
# and, with the persisted vectorstore returning real chunks, answers fine.
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.2
TOP_K = 3


# ---------- Page config ----------
st.set_page_config(
    page_title="CloudNest Support",
    page_icon="☁️",
    layout="centered",
)


# ---------- Secrets ----------
# Streamlit Cloud injects [secrets] into st.secrets; locally we use .streamlit/secrets.toml
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "⚠️ GROQ_API_KEY is not set. "
        "Add it to `.streamlit/secrets.toml` locally, or to the Streamlit Cloud "
        "secrets manager when deployed."
    )
    st.stop()


# ---------- Cached resources (build once per session) ----------
@st.cache_resource(show_spinner="Loading knowledge base and embedding model (first run takes ~30s)...")
def get_vectorstore():
    return build_default_vectorstore()


@st.cache_resource(show_spinner="Initializing agent...")
def get_agent(_vectorstore, model_name: str, temperature: float, k: int):
    """Note the leading underscore on _vectorstore — tells Streamlit not to hash it."""
    retriever = _vectorstore.as_retriever(search_kwargs={"k": k})
    return build_agent(retriever, model_name=model_name, temperature=temperature)


# ---------- Sidebar ----------
with st.sidebar:
    st.title("☁️ CloudNest Support")
    st.caption("Your virtual assistant for accounts, billing, and troubleshooting.")

    st.divider()
    if st.button("🔄 Reset conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("thread_id", None)
        st.rerun()


# ---------- Build resources ----------
vectorstore = get_vectorstore()
agent = get_agent(vectorstore, MODEL_NAME, TEMPERATURE, TOP_K)


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    # A unique thread id per session enables multi-turn memory in the agent
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())


# ---------- Header ----------
st.title("CloudNest Support")
st.caption("Ask me about pricing, sharing, refunds, troubleshooting, or anything CloudNest.")


# ---------- Render conversation history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_calls"):
            with st.expander(f"🔧 Tools used ({len(msg['tool_calls'])})"):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['name']}({tc['args']})", language="python")


# ---------- Chat input ----------
if prompt := st.chat_input("Ask a question..."):
    # Render the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and render the assistant message
    with st.chat_message("assistant"):
        placeholder = st.empty()
        tool_calls_seen = []

        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            # Stream agent steps so we can show tool-call activity in real time
            final_text = ""
            for event in agent.stream(
                {"messages": [("user", prompt)]},
                config=config,
                stream_mode="values",
            ):
                last_msg = event["messages"][-1]
                # Capture tool calls as they happen
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_calls_seen.append({"name": tc["name"], "args": tc["args"]})
                        placeholder.info(f"🔍 Calling `{tc['name']}`...")
                # Capture the final assistant text
                if last_msg.type == "ai" and last_msg.content:
                    final_text = last_msg.content

            placeholder.markdown(final_text or "_(no response)_")

            if tool_calls_seen:
                with st.expander(f"🔧 Tools used ({len(tool_calls_seen)})"):
                    for tc in tool_calls_seen:
                        st.code(f"{tc['name']}({tc['args']})", language="python")

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_text,
                "tool_calls": tool_calls_seen,
            })

        except Exception as e:
            err = f"⚠️ Something went wrong: `{type(e).__name__}: {e}`"
            placeholder.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
