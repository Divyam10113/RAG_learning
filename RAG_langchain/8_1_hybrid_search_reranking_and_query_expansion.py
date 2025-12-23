import streamlit as st
import os
import logging
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- ADVANCED RETRIEVAL IMPORTS ---
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

st.set_page_config(page_title="RAG Teacher Bot", page_icon="🤖")
st.title("🤖 Chat with Artemis Handbook (Ultimate Edition)")

# Define paths
PDF_FILE = "8_2_doc.pdf"  # Make sure this file exists in your folder!
PERSIST_DIR = "chroma_db_storage_reranked"

# --- 1. SETUP THE BRAIN (Cached) ---
@st.cache_resource
def initialize_rag_chain():
    # 1. SETUP: Load Data and Build Indices
    # We use 'all-MiniLM-L6-v2' for vectors because it's fast and effective
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We MUST load the PDF to build the BM25 (Keyword) index every time.
    # (BM25 is an in-memory index, unlike Chroma which saves to disk)
    if not os.path.exists(PDF_FILE):
        st.error(f"File {PDF_FILE} not found! Please check the filename.")
        st.stop()
        
    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 2. SETUP VECTOR DATABASE (Chroma)
    # Checks if DB exists on disk to save time, otherwise builds it
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=PERSIST_DIR)
    
    # --- STEP 1: HYBRID RETRIEVER (Vector + Keyword) ---
    
    # A. Keyword Retriever (BM25)
    # Finds exact matches (Good for IDs, specific terms)
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10 
    
    # B. Vector Retriever (Chroma)
    # Finds conceptual matches (Good for meanings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # C. Ensemble Retriever (Hybrid)
    # Combines both. Weights: 0.5 for Vector, 0.5 for Keyword.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], 
        weights=[0.5, 0.5]
    )

    # --- STEP 2: MULTI-QUERY (Expansion) ---
    
    # Setup LLM for generating queries and answering
    # Using 'gemini-1.5-flash' as it is the standard stable model.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    
    # The Brainstormer: Generates 3 variations of the user's question
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever, # Feeds into Hybrid
        llm=llm
    )

    # --- STEP 3: RE-RANKER (The Judge) ---
    
    # Filters the large pool of results (from Hybrid+MultiQuery) down to the best 5
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever 
    )
    
    # --- FINAL CHAIN SETUP ---
    
    # 1. History Contextualization (Handling "it", "that", etc.)
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        final_retriever, 
        contextualize_q_prompt
    )
    
    # 2. Answer Generation (Research Assistant Persona)
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

# --- 3. CHAT UI ---
user_input = st.chat_input("Ask about the paper...")

if user_input:
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # B. Add User Message to History
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # C. "Thinking" Status Indicator
    with st.status("🧠 Thinking...", expanded=True) as status:
        st.write("🚀 Generating alternative queries (Multi-Query)...")
        st.write("🔍 Searching database (Hybrid: Vector + Keyword)...")
        st.write("⚖️ FlashRank is re-ranking results...")
        status.update(label="Answer found!", state="complete", expanded=False)

    # D. Stream the Response using st.write_stream
    with st.chat_message("assistant"):
        def stream_generator():
            stream = rag_chain.stream({
                "input": user_input, 
                "chat_history": st.session_state.chat_history
            })
            for chunk in stream:
                if "answer" in chunk:
                    yield chunk["answer"]
        
        response = st.write_stream(stream_generator())
    
    # E. Save AI Answer to History
    st.session_state.chat_history.append(AIMessage(content=response))