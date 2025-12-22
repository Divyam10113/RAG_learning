import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import Client
from dotenv import load_dotenv

load_dotenv()
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Connecting to the brain...")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

response = llm.invoke("Hello! Are you ready to teach me RAG?")
print("\nResponse from Gemini:")
print(response.content)