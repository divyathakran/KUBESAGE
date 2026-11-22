import logfire

from app.agents.state import AgentState
from app.gateway.client import (
    litellm_client,
    extract_cache_status,
    PRIMARY_MODEL
)


def generate_node(state: AgentState):
    """
    Synthesizes a response using technical documentation
    and conversation history.

    Uses the native OpenAI-compatible LiteLLM client
    to access response headers and detect cache hits.
    """

    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":

        logfire.info("Generating conversational response using memory.")

        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:

        logfire.info("Generating technical RAG response.")

        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit model limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("LLM Synthesis"):

        try:
            raw = litellm_client.chat.completions.with_raw_response.create(
                model=PRIMARY_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                extra_body={
                    "metadata": {
                        "tags": ["responder"]
                    }
                }
            )

            # Check whether LiteLLM served this response from cache.
            cache_status = extract_cache_status(raw)

            # Convert the raw HTTP response into a completion object.
            completion = raw.parse()

            answer = completion.choices[0].message.content

            logfire.info(
                "Gateway cache {status}",
                status=cache_status
            )

            logfire.info("Response synthesised via LiteLLM.")

            return {
                "final_answer": answer,
                "status": "Response generated",
                "plan": state["plan"],
                "messages": [
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise