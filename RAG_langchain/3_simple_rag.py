import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated to match your working code
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# --- CONFIGURATION ---
load_dotenv()
# 1. THE DATA (Our "Textbook")
# In real life, this comes from a PDF. Today, we fake it to test the pipeline.
text_data = [
    "The secret project is called 'Apollo'.",
    "Apollo's launch date is set for July 15th.",
    "The project manager for Apollo is Sarah Connor.",
    "The budget for Apollo is $50 million.",
    "This is a distractor sentence about bananas.",
    "Another random sentence about the weather."
]

# 2. THE MEMORY (Vector Database)
# We turn the text into numbers and store them in ChromaDB
print("1. Indexing data...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# This creates a temporary in-memory database
vectorstore = Chroma.from_texts(
    texts=text_data,
    embedding=embeddings
)

# We set up a "Retriever" to fetch the top 1 most relevant sentence
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) 

# 3. THE BRAIN (LLM)
# Using the model you confirmed works
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# 4. THE PROMPT
# This is the instruction we send to the LLM
template = """
Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. THE PIPELINE (Chain)
# This is the "LangChain" magic:
# Retriever -> (finds text) -> Prompt -> (adds text) -> LLM -> (answers)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. ASK A QUESTION
# Note: The LLM doesn't know "Sarah Connor" is the manager. It MUST read the text to know.
my_question = "Who is the project manager?"
print(f"\nQuestion: {my_question}")

print("2. Retrieving answer...")
response = rag_chain.invoke(my_question)

print(f"Answer: {response}")

my_question = "How much is her budget?"
print(f"\nQuestion: {my_question}")

print("2. Retrieving answer...")
response = rag_chain.invoke(my_question)

print(f"Answer: {response}")