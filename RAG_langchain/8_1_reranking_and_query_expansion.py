import streamlit as st
import os
from flashrank import Ranker, RerankRequest
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
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_compressors import FlashrankRerank
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="RAG Teacher Bot", page_icon="🤖")
st.title("🤖 Chat with Artemis Handbook")

# Define paths
PDF_FILE = "8_2_doc.pdf"  # Ensure this matches your actual PDF name
PERSIST_DIR = "chroma_db_storage_reranked"

# --- 1. SETUP THE BRAIN (Cached) ---
@st.cache_resource
def initialize_rag_chain():
    # A. Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # B. Load DB (or create if missing)
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        if not os.path.exists(PDF_FILE):
            st.error(f"File {PDF_FILE} not found! Please check the filename.")
            st.stop()
            
        loader = PyPDFLoader(PDF_FILE)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=PERSIST_DIR)
    
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. The Judge (Flashrank): It will sort the 10 docs and pick the top 3
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    
    # 3. Compression Retriever: Combines the Base + The Judge
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    # ---------------------------------------
    
    # C. Setup LLM (Using standard flash model)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=compression_retriever,
        llm=llm
    )
    
    # D. "History Aware" Retriever
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, multi_query_retriever, contextualize_q_prompt)
    
    # E. Answer Question Chain (Artemis Persona)
    # F. Answer Question Chain (Research Assistant Persona)
    qa_system_prompt = """You are an expert AI Research Assistant specializing in Deep Learning and NLP.
    Your goal is to explain complex technical concepts from the research paper clearly and accurately.
    
    RULES:
    1. Answer strictly based on the provided context.
    2. If the context contains mathematical formulas (LaTeX), interpret them for the user in plain English if possible.
    3. When explaining architectures (like the Encoder or Decoder), be specific about layers and dimensions (e.g., d_model=512).
    4. If the answer is not in the paper, state "The paper does not provide information on this."
    
    CONTEXT:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Start the chain
rag_chain = initialize_rag_chain()

# --- 2. MANAGE CHAT HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- 3. CHAT UI (THE FIX) ---
user_input = st.chat_input("Ask about the paper...")

if user_input:
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # B. Add User Message to History
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # We place this OUTSIDE the assistant message so it looks like a system status
    with st.status("🧠 Thinking...", expanded=True) as status:
        st.write("🚀 Generating alternative queries...")
        st.write("🔍 Searching vector database...")
        st.write("⚖️ FlashRank is re-ranking top 10 results...")
        # We can't actually "pause" the code here to show these steps one by one 
        # without advanced callbacks, so this will show up quickly just before the answer.
        status.update(label="Answer found!", state="complete", expanded=False)

    # C. Stream the Response using st.write_stream
    with st.chat_message("assistant"):
        # We use a generator function to pull text chunks one by one
        def stream_generator():
            stream = rag_chain.stream({
                "input": user_input, 
                "chat_history": st.session_state.chat_history
            })
            for chunk in stream:
                # We filter for the 'answer' key which contains the text
                if "answer" in chunk:
                    yield chunk["answer"]
        
        # This single line handles the streaming animation automatically
        response = st.write_stream(stream_generator())
    
    # D. Save AI Answer to History
    st.session_state.chat_history.append(AIMessage(content=response))