import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyBHTI-j06hMCz59PPXhUvD2Jbp-50JCpQg"

print("Connecting to the brain...")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

response = llm.invoke("Hello! Are you ready to teach me RAG?")
print("\nResponse from Gemini:")
print(response.content)