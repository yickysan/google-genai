from audit_agent.agent_graph import chat_graph
from audit_agent.retriever import add_document_to_vectorstore



def main():
    # config = {"recursion_limit": 1000}
    # chat_graph.invoke({"messages": []}, config)

    add_document_to_vectorstore()

    
    


if __name__ == "__main__":
    main()
