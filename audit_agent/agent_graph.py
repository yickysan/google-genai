from langgraph.graph import StateGraph, START

from .agent import (AuditAssistantState,
                    audit_assistant,
                    human_node,
                    router_decison,
                    router_node,
                    policy_node,
                    search_tool_node,
                    maybe_exit_human_node
                    ) 


graph_builder = StateGraph(AuditAssistantState)

graph_builder.add_node("audit_assistant", audit_assistant)
graph_builder.add_node("human", human_node)
graph_builder.add_node("router", router_node)
graph_builder.add_node("policy_expert", policy_node)
graph_builder.add_node("search_web", search_tool_node)
graph_builder.add_edge(START, "audit_assistant")
graph_builder.add_edge("audit_assistant", "human")
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_conditional_edges("router", router_decison)
# graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# graph_builder.add_edge("tools", "chatbot")
chat_graph = graph_builder.compile()
