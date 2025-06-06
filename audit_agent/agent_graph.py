from langgraph.graph import StateGraph, START

from .agent import (AuditAssistantState,
                    audit_assistant,
                    human_node,
                    router_decison,
                    router_node,
                    retrieve,
                    search,
                    maybe_exit_human_node
                    ) 


graph_builder = StateGraph(AuditAssistantState)

graph_builder.add_node("audit_assistant", audit_assistant)
graph_builder.add_node("human", human_node)
graph_builder.add_node("router", router_node)
graph_builder.add_node("retriever", retrieve)
graph_builder.add_node("search_web", search)
graph_builder.add_edge(START, "audit_assistant")
graph_builder.add_edge("retriever", "audit_assistant")
graph_builder.add_edge("search_web", "audit_assistant")
graph_builder.add_edge("audit_assistant", "human")
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_conditional_edges("router", router_decison,
                                    {"vectorstore": "retriever",
                                     "search_web": "search_web",
                                     "audit_assistant": "audit_assistant"})
# graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# graph_builder.add_edge("tools", "chatbot")
chat_graph = graph_builder.compile()
