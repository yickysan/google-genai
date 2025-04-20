from langgraph.graph import StateGraph, START

from audit_agent.agent_graph import chat_graph


def main():
    config = {"recursion_limit": 1000}
    #chat_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
    chat_graph.invoke({"messages": []}, config)

    # add_document_to_vectorstore(doc_path)
    # res = retriever("Who is a PEP?")
    # print(res)
    
    


if __name__ == "__main__":
    main()
