import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore # Stores the "Big" chunks in RAM
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PDF_FILE = "8_2_doc.pdf" # Use your Transformer paper
PERSIST_DIR = "chroma_db_parent_child"

print("--- 1. LOADING DATA ---")
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

# --- 2. DEFINE THE TWO SPLITTERS ---

# The "Parent" Splitter (Large Context)
# This keeps big sections together (e.g., 2000 characters)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# The "Child" Splitter (Precision Search)
# This chops parents into tiny search keys (e.g., 400 characters)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# --- 3. SETUP STORAGE ---

# The Vector Store (Chroma) will hold the SMALL "Child" chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="split_parents", 
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR
)

# The Doc Store (Memory) will hold the LARGE "Parent" chunks
# When a child is found, the retriever looks up the ID here to get the full text.
store = InMemoryStore()

# --- 4. THE MAGIC: PARENT-CHILD RETRIEVER ---
print("--- 2. INDEXING (This might take a moment) ---")
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# We add documents to the retriever, and it handles the splitting & linking automatically
retriever.add_documents(docs)

# --- 5. TEST IT ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

print("\n--- SYSTEM READY ---\n")

# A question that requires context (comparing two things)
question = "Compare the BLEU score of the base model vs the big model on EN-DE."
print(f"Question: {question}")

result = qa_chain.invoke(question)
print(f"\nAnswer: {result['result']}")

# --- 6. DEBUG: PROVE IT WORKED ---
print("\n[Under the Hood]")
# Let's see what the retriever actually found
retrieved_docs = retriever.invoke(question)
print(f"Retrieved {len(retrieved_docs)} Parent Documents.")
print(f"First Parent Document Length: {len(retrieved_docs[0].page_content)} characters.")
print(f"Preview: {retrieved_docs[0].page_content[:]}")