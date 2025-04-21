import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama


load_dotenv()


GOOGLEAPI = os.getenv("GOOGLEAPI")
os.environ["GOOGLE_API_KEY"] = GOOGLEAPI

GOOGLE= ChatGoogleGenerativeAI(
    # temperature=0,
    model="gemini-2.0-flash",
    # max_tokens=300,
    # top_p=0.95,
    # top_k=40,
    # streaming=True,
)

OLLAMA = ChatOllama(
    model="llama3.2:3b",
    temperature=3,
    max_tokens=300,
    # top_p=0.95,
    # top_k=40,
    streaming=True,
)

LLM = GOOGLE