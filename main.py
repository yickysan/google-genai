import os
from typing import Annotated, TypedDict, Literal, Iterable

from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()


GOOGLEAPI = os.getenv("GOOGLEAPI")
os.environ["GOOGLE_API_KEY"] = GOOGLEAPI

class Agent(TypedDict):
    """
    A simple agent with state management.
    """

    messages: Annotated[list, add_messages]
    query: list[str]
    finished: bool

SYS_PROMPT = (
    "system",
    "You are a helpful assistant to help the internal audit staff of FSDH achieve their goals"
    "Auditors will come to you with questions about the audit process that should be tested for an audit"
    "You will answer them to the best of your ability and give reasons for your answers"
    "They will also ask you aabout the bank's policies, to answer call_get_policy (this is shown to you and not the user)"
    "If you don't know the answer, say 'I don't know'"
    "You are allowed to search the web for answers, to do this use search_web"
    "Always confirm with auditors if they have additional information to add to the question"
    "If they do, ask them to provide it"
    "If they don't, ask them to provide a summary of the question"
)

WELCOME_PROMPT = (
    "Welcome to the FSDH Audit Assistant "
    "I am here to help you with your audit questions. "
    "Please ask me anything about the audit process or the bank's policies.\n"
    "Type exit or quit to end the conversation."
)

llm = ChatGoogleGenerativeAI(
    # temperature=0,
    model="gemini-2.0-flash",
    # max_tokens=300,
    # top_p=0.95,
    # top_k=40,
    # streaming=True,
)

search_tool = DuckDuckGoSearchResults()

def chatbot(state: Agent) -> Agent:

    if state["messages"]:
        message_history = [SYS_PROMPT] + state["messages"]
        out = llm.invoke(message_history)

    else:
        out = AIMessage(content=WELCOME_PROMPT)

    return state | {"messages": [out]}


def human_node(state: Agent) -> Agent:
    last_message = state["messages"][-1]

    user_input = input(
        f"Assistant: {last_message.content}\n"
        "User: "
    )
    
    if user_input.lower() in ["exit", "quit", "stop"] :
        state["finished"] = True
   
    return state | {"messages": [("user", user_input)]}


def maybe_exit_human_node(state: Agent) -> Literal["chatbot", "__end__"]:
    if state.get("finished", False):
        return END
    else:
        return "chatbot"
    
graph_builder = StateGraph(Agent)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human", human_node)
graph_builder.add_node("search_web", search_tool.run)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
chat_graph = graph_builder.compile()



def main():
    config = {"recursion_limit": 1000}

    chat_graph.invoke({"messages": []}, config)


if __name__ == "__main__":
    main()
