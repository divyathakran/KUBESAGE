from app.agents.state import AgentState
import logfire
from app.gateway.client import get_langchain_llm

llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = f"""
You are an intelligent Assistant Planner.

Your supported domains are:
1. Kubernetes
2. Intel technologies
3. Computer Networking

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{user_message}"

Classify the latest message:

1. CONVERSATIONAL:
Greetings or questions answerable using conversation history.

2. TECHNICAL:
Questions related to Kubernetes, Intel, or Networking
that require technical documentation.

3. OUT_OF_SCOPE:
Questions unrelated to the supported domains.

Output ONLY one of:
CONVERSATIONAL
OUT_OF_SCOPE
OR a refined technical search query.
"""
    
    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    if decision == "OUT_OF_SCOPE":
        return {
            "current_query": "OUT_OF_SCOPE",
            "status": "Question outside supported domains.",
            "plan": ["Intent: Out of Scope", "Retrieval: Skipped"],
            "documents": []
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }