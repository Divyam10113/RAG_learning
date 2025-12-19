import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PDF_FILE = "4_1_mydocument.pdf" # Use your actual PDF filename
PERSIST_DIR = "chroma_db_storage"

# 1. SETUP (Load DB or Build it - Optimized logic from Phase 4)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Loading existing database...")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    print("Building database...")
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=PERSIST_DIR)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# --- PHASE 5: THE CONVERSATIONAL BRAIN ---

# Step A: The "Reformulation" Prompt
# This tells the LLM: "Given the chat history, rewrite the new question to stand alone."
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Step B: The History-Aware Retriever
# This is a chain that sits BEFORE the main RAG chain.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step C: The Answer Prompt (Standard RAG)
qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
\n\n
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Step D: The Final Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- THE LOOP ---
chat_history = []  # <--- This List stores our memory!

print("\n--- CONVERSATIONAL RAG READY ---")
print("Ask questions! (type 'exit' to quit)")

while True:
    query = input("\nYour Question: ")
    if query.lower() == "exit":
        break
    
    # We pass the 'chat_history' along with the query
    result = rag_chain.invoke({"input": query, "chat_history": chat_history})
    
    print(f"Answer: {result['answer']}")
    
    # Update Memory
    # We append the User's question and the AI's answer to the history list
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result['answer']))