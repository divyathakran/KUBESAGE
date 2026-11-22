from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node

def out_of_scope_node(state: AgentState):

    answer = (
        "I'm KubeSage, an Enterprise IT Assistant. "
        "I can help with Kubernetes, Intel technologies, "
        "and enterprise networking. Please ask me something "
        "related to these topics."
    )

    return {
        "messages": [{
            "role": "assistant",
            "content": answer
        }],
        "final_answer": answer,
        "status": "Request rejected: Outside supported domains.",
        "plan": ["Intent: Out of Scope", "Retrieval: Skipped"],
        "documents": []
    }

# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("out_of_scope", out_of_scope_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):

    query = state["current_query"]

    if query == "CONVERSATIONAL":
        return "responder"

    if query == "OUT_OF_SCOPE":
        return "out_of_scope"

    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
        "out_of_scope": "out_of_scope"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
workflow.add_edge("out_of_scope", END)


# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)