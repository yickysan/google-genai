from audit_agent.agent_graph import chat_graph



def main():
    config = {"recursion_limit": 1000}
    chat_graph.invoke({"messages": []}, config)

    
    


if __name__ == "__main__":
    main()
