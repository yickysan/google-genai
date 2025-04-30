from pathlib import Path

from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from . import LLM


docs = [Path("C:/Users/aeniatorudabo/Documents/7.3 - AML CFT CPF Policy.pdf"),
        Path("C:/Users/aeniatorudabo/Downloads/FSDH IA Standards and Procedures Manual - Bank.pdf")]


connection = SQLiteVec.create_connection(db_file="policies.db")

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_document(files: str | Path | list[str | Path]) -> list[str]:
    loader = UnstructuredLoader(files)
    return loader.load()

def add_document_to_vectorstore() -> SQLiteVec:

    doc = load_document(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #embeddings2 = OllamaEmbeddings(model="nomic-embed-text")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

    chunks = text_splitter.split_documents(doc)

    vectorstore = SQLiteVec.from_documents(chunks,
                                            embedding=embeddings,
                                            table = "policy",
                                            db_file="policies.db")

    return vectorstore




    

def retriever(query: str) -> list[str]:

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
    # prompt = f"""
    # Answer the question based ONLY on the following context:
    # {context}
    # Question: {query}
    # Give your output in markdown format
    # """
    # response = LLM.invoke(prompt)
    return context