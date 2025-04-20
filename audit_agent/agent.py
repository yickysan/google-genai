from typing import Annotated, TypedDict, Literal

from langgraph.graph.message import add_messages
from langgraph.graph import END
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults

from . import LLM
from .router import Router
from .retriever import retriever


search_tool = DuckDuckGoSearchResults()

llm_router = LLM.with_structured_output(Router)

SYS_PROMPT = (
    "system",
    "You are a helpful assistant to help the internal audit staff achieve their goals"
    "Auditors will come to you with questions about the audit process that should be tested for an audit"
    "You will answer them to the best of your ability and give reasons for your answers"
    "They will also ask you aabout the bank's policies, you will answer by retrieving the policy from a vector database"
    "If you don't know the answer, say 'I don't know'"
    "You are allowed to search the web for answers. Review answers getting from web search and combine multiple results into a final answer"
    "Include sources for your final output of the websearch"
    "Always confirm with auditors if they have additional information to add to the question"
    "If they do, ask them to provide it"
    "If they don't, ask them to provide a summary of the question"
    "Do not show tool names to the user"
    "Do not show the tool code to the user, only the final answer"
)

WELCOME_PROMPT = (
    "Welcome to the Audit Assistant "
    "I am here to help you with your audit questions.\n"
    "Please ask me anything about the audit process or the bank's policies.\n"
    "Type exit or quit to end the conversation."
)


class AuditAssistantState(TypedDict):
    """
    A simple agent with state management.
    """

    messages: Annotated[list, add_messages]
    decision: str
    finished: bool


def audit_assistant(state: AuditAssistantState) -> AuditAssistantState:

    if state["messages"]:
        message_history = [SYS_PROMPT] + state["messages"]
        output = LLM.invoke(message_history)

    else:
        output = AIMessage(content=WELCOME_PROMPT)

    return state | {"messages": [output]}

def search_tool_node(state: AuditAssistantState) -> AuditAssistantState:
    """
    Call the search to search the web
    """
    # Get the last message from the state
    last_message = state["messages"][-1]

    # Get the query from the last message
    query = last_message.content

    # Call the search tool with the query
    response = search_tool.invoke(query)

    # Add the response to the state
    return state | {"messages": [("tool", response)]}


def human_node(state: AuditAssistantState) -> AuditAssistantState:
    last_message = state["messages"][-1]

    user_input = input(
        f"Assistant: {last_message.content}\n"
        "User: "
    )
    
    if user_input.lower() in ["exit", "quit", "stop"] :
        state["finished"] = True
   
    return state | {"messages": [("user", user_input)]}


def router_node(state: AuditAssistantState) -> AuditAssistantState:
    """
    Call the router to route between states
    """
    # Get the last message from the state
    last_message = state["messages"][-1]

    # Call the router with the query
    decision= llm_router.invoke(
        [SystemMessage(content="Route the input to audit_assistant, search_web or policy_expert based on the user input"),
         HumanMessage(content=last_message.content),
        ]
        )

    # Add the response to the state
    return state | {"decision": decision.step}

def maybe_exit_human_node(state: AuditAssistantState) -> Literal["audit_assistant", "__end__"]:
    if state.get("finished", False):
        return END
    else:
        return "audit_assistant"
    
def router_decison(state: AuditAssistantState) -> str:
    """
    Route the state based on the decision
    """
    # Get the decision from the state
    decision = state["decision"]

    if decision == "audit_assistant":
        return "audit_assistant"
    elif decision == "search_web":
        return "search_web"
    elif decision == "policy_expert":
        return "policy_expert"
    

def maybe_route_to_tools(state: AuditAssistantState) -> Literal["tools", "human"]:
    """Route between human or tool nodes, depending if a tool call is made."""
    if not (messages := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    # Only route based on the last message.
    message = messages[-1]

    # When the chatbot returns tool_calls, route to the "tools" node.
    if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
        return "tools"
    else:
        return "human"
    
def policy_node(state: AuditAssistantState) -> AuditAssistantState:
    """
    Call the retriever tool to get the policy
    """
    # Get the last message from the state
    last_message = state["messages"][-1]

    # Get the query from the last message
    query = last_message.content

    # Call the retriever tool with the query
    response = retriever.invoke(query)

    # Add the response to the state
    return state | {"messages": response}



# tool_node = ToolNode(tools)
