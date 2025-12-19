import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PDF_FILE = "4_1_mydocument.pdf"  # Make sure this file exists!
PERSIST_DIR = "chroma_db_storage"  # Folder where we save the database

print("1. Initializing Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Found existing database! Loading...")
    # CODE TO LOAD EXISTING DB GOES HERE
    # Hint: vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function= embeddings
    )
else:
    print("No database found. Building from PDF...")
    # CODE TO READ PDF AND BUILD DB GOES HERE
    print(f"1. Loading {PDF_FILE}...")
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"Could not find {PDF_FILE}. Please move a PDF into this folder.")

    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    print(f"   Loaded {len(docs)} pages.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"   Split into {len(splits)} chunks.")

    # 3. EMBED & STORE (The "Memory" Phase)
    print("3. Creating Vector Database (this takes time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # persist_directory tells Chroma to save to disk
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR 
    )
    print("   Database saved to disk!")




# 2. CHUNK THE TEXT (The "Slicing" Phase)
# We split the text into 1000-character chunks with 200 overlap

# 4. SETUP THE RETRIEVAL ENGINE
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

# 5. CONNECT THE LLM
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# 6. CREATE THE CHAT CHAIN
# RetrievalQA is a pre-built chain that handles the "Prompt + Context" logic for us
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "Stuff" means "stuff all the context into the prompt"
    retriever=retriever,
    return_source_documents=True # Show us WHICH page the answer came from
)

# 7. THE INTERACTIVE LOOP
print("\n--- RAG SYSTEM READY ---")
print("Ask questions about your PDF (type 'exit' to quit)")

while True:
    query = input("\nYour Question: ")
    if query.lower() == "exit":
        break
    
    # Run the query
    result = qa_chain.invoke({"query": query})
    
    print(f"\nAnswer: {result['result']}")
    
    # Bonus: Show sources
    print("\n[Sources Used:]")
    for doc in result['source_documents']:
        print(f"- Page {doc.metadata.get('page', '?')}: {doc.page_content[:50]}...")