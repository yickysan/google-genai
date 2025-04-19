import os
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from dotenv import load_dotenv

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import tool
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


doc_path = Path("C:/Users/aeniatorudabo/Documents/7.3 - AML CFT CPF Policy.pdf")

connection = SQLiteVec.create_connection(db_file="vec.db")

GOOGLEAPI = os.getenv("GOOGLEAPI")
os.environ["GOOGLE_API_KEY"] = GOOGLEAPI

LLM= ChatGoogleGenerativeAI(
    # temperature=0,
    model="gemini-2.0-flash",
    # max_tokens=300,
    # top_p=0.95,
    # top_k=40,
    # streaming=True,
)

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_document(file: str | Path) -> list[str]:
    loader = UnstructuredLoader(file)
    return loader.load()

# def add_document_to_vectorstore(file: str | Path) -> SQLiteVec:

#     doc = load_document(file)

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

#     chunks = text_splitter.split_documents(doc)

#     vectorstore = SQLiteVec.from_documents(chunks,
#                                             embedding=embeddings,
#                                             persist_directory="vectorstore.db",
#                                             collection_name="audit_policy",)

#     return vectorstore



class Agent(TypedDict):
    """
    A simple agent with state management.
    """

    messages: Annotated[list, add_messages]
    query: list[str]
    finished: bool

SYS_PROMPT = (
    "system",
    "You are a helpful assistant to help the internal audit staff achieve their goals"
    "Auditors will come to you with questions about the audit process that should be tested for an audit"
    "You will answer them to the best of your ability and give reasons for your answers"
    "They will also ask you aabout the bank's policies, to answer use retriever_tool to get the answer (this is shown to you and not the user)"
    "the policy documents have already been loaded into the vectorstore"
    "If you don't know the answer, say 'I don't know'"
    "You are allowed to search the web for answers, to do this use search_web"
    "Always confirm with auditors if they have additional information to add to the question"
    "If they do, ask them to provide it"
    "If they don't, ask them to provide a summary of the question"
    "Do not show tool names to the user"
)

WELCOME_PROMPT = (
    "Welcome to the Audit Assistant "
    "I am here to help you with your audit questions. "
    "Please ask me anything about the audit process or the bank's policies.\n"
    "Type exit or quit to end the conversation."
)



@tool
def retriever_tool(query: str) -> list[str]:

    """
    Call the retriever tool to get the policy
    """
    vector_db = SQLiteVec(table="langchain", embedding=EMBEDDINGS, connection=connection)

    QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    Your task is to generate five different versions
    of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user
    overcome some of the limitations of the distance-based similarity search.
    Provide alternative questions seperated by new lines.
    Original question: {question}""",

    )

    retriever = MultiQueryRetriever.from_llm(
                    retriever=vector_db.as_retriever(),
                    llm=LLM,
                    prompt=QUERY_PROMPT,
                    parser_key="lines"
                    #verbose=True,
                    )

    docs = retriever.invoke(query)

    # Generate a response based on the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Answer the question based ONLY on the following context:
    {context}
    Question: {query}
    """
    response = LLM.invoke(prompt)
    return response.content

search_tool = DuckDuckGoSearchResults()

tools = [retriever_tool]

llm_with_tool = LLM.bind_tools(
    tools=tools
    )





def chatbot(state: Agent) -> Agent:

    if state["messages"]:
        message_history = [SYS_PROMPT] + state["messages"]
        out = llm_with_tool.invoke(message_history)

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
    
def maybe_route_to_tools(state: Agent) -> Literal["tools", "human"]:
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


tool_node = ToolNode(tools)

    
graph_builder = StateGraph(Agent)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("search_web", search_tool.run)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "human")
graph_builder.add_conditional_edges("human", maybe_exit_human_node)
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
graph_builder.add_edge("tools", "chatbot")
chat_graph = graph_builder.compile()



def main():
    config = {"recursion_limit": 1000}
    # add_document_to_vectorstore(doc_path)
    # res = retriever_tool("Who is a PEP?")
    # print(res)

    chat_graph.invoke({"messages": []}, config)
    


if __name__ == "__main__":
    main()
