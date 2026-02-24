# =============================================================================
# RAG (Retrieval Augmented Generation) Practice with LangChain & ChromaDB
# =============================================================================
# LangChain, LangGraph, Pinecone, and Flowise
# OpenAI embeddings are advanced AI models that 
# transform text into numerical vectors, mapping semantic meaning to capture 
# relationships between concepts, rather than just keywords. 
# They power applications like semantic search, content recommendations, 
# and RAG (Retrieval Augmented Generation) by enabling fast, 
# context-aware similarity searches.

# --- IMPORTS ---

# ChatOpenAI: A LangChain wrapper around OpenAI's chat models (e.g., GPT-3.5, GPT-4).
#   It lets you send messages to the model and get responses, with LangChain's
#   unified interface (invoke, stream, batch, etc.).
# OpenAIEmbeddings: A LangChain wrapper that calls OpenAI's embedding API to convert
#   text strings into high-dimensional numerical vectors (lists of floats).
#   These vectors capture the semantic meaning of the text so that similar
#   texts end up with similar vectors, enabling similarity search.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# langchain-chroma is the official integration 
# package connecting the LangChain framework to ChromaDB, 
# an open-source vector database designed for AI application development. 
# It enables efficient semantic search, document storage, 
# and retrieval by managing high-dimensional embeddings 
# directly within LangChain, allowing developers to 
# create persistent, local, or server-based vector stores.

# --- EXAMPLE BLOCK (commented out as a docstring) ---
# This triple-quoted block is a multi-line string that acts as a commented-out
# example showing how to use Chroma with persistence (saving to disk).
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Create embeddings
embedding_function = OpenAIEmbeddings()

# Initialize Chroma with a persistent directory
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_function,
    persist_directory="./chroma_db"
)

# Add documents
vector_store.add_texts(["LangChain is a framework for LLMs", "Chroma is a vector database"])

# Perform a similarity search
results = vector_store.similarity_search("What is Chroma?")
print(results[0].page_content)
"""

# --- ACTIVE CODE STARTS HERE ---

# Chroma: LangChain's integration with ChromaDB, an open-source vector database.
#   It stores document embeddings (vectors) and allows fast similarity search.
#   Here it runs in-memory (no persist_directory), so data is lost when the script ends.
from langchain_chroma import Chroma

# RecursiveCharacterTextSplitter: A text splitter that breaks large text into
#   smaller chunks. It tries to split on natural boundaries (paragraphs, sentences,
#   words) recursively, keeping chunks under a specified size. This is critical
#   because embedding models and LLMs have token limits, and smaller chunks
#   improve retrieval precision.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document: A simple data class from LangChain that wraps text content.
#   It has two main fields:
#     - page_content (str): the actual text
#     - metadata (dict): optional key-value pairs (e.g., source, page number)
#   LangChain tools expect data in this Document format.
from langchain_core.documents import Document

# load_dotenv: Reads a .env file from the current directory and loads its
#   key-value pairs as environment variables (e.g., OPENAI_API_KEY).
#   This keeps sensitive API keys out of source code.
from dotenv import load_dotenv

# Actually load the .env file now. After this call, os.environ["OPENAI_API_KEY"]
# (and any other variables in .env) will be available to the libraries that need them.
# OpenAI client libraries automatically look for the OPENAI_API_KEY env variable.
load_dotenv()

# --- STEP 1: Define the raw knowledge base ---
# This is the raw text that represents our "knowledge base" — the information
# we want the RAG system to be able to answer questions about.
# In a real application, this could come from files, databases, web scraping, etc.
text = """
Our refund policy allows returns within 30 days. 
Our premium plan costs $50 per month. 
We offer 24/7 customer support via email and chat.
"""

# --- STEP 2: Wrap the text in a Document object ---
# LangChain's text splitters and vector stores work with Document objects.
# We wrap our raw text string into a Document so it can be processed by the pipeline.
# The list can contain multiple Documents if you have multiple sources.
documents = [Document(page_content=text)]

# --- STEP 3: Split the text into smaller chunks ---
# Create a text splitter with:
#   chunk_size=100   → each chunk will be at most 100 characters long
#   chunk_overlap=20 → consecutive chunks share 20 characters of overlap
# The overlap ensures that if a relevant sentence sits at a chunk boundary,
# it won't be completely lost — part of it will appear in both chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# split_documents() takes a list of Documents and returns a new list of smaller
# Document objects, each containing a chunk of the original text.
# For our ~150-character text, this will produce roughly 2-3 chunks.
chunks = text_splitter.split_documents(documents)

# --- STEP 4: Create the embedding function ---
# Initialize OpenAI's embedding model. "text-embedding-3-small" is a compact,
# cost-effective model that converts text into 1536-dimensional vectors.
# Each chunk of text will be sent to this model to get its vector representation.
# These vectors numerically encode the semantic meaning of each chunk.
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# --- STEP 5: Create the vector store and index the chunks ---
# Chroma.from_documents() does THREE things in one call:
#   1. Takes each chunk's page_content and sends it to the embedding function
#      to generate a vector (list of floats) for that chunk.
#   2. Stores each vector along with its original text in ChromaDB's in-memory database.
#   3. Builds an index so that similarity searches can be performed efficiently.
# After this, the vector_store contains all chunks as searchable vectors.
vector_store = Chroma.from_documents(
    documents=chunks,        # The list of Document chunks to embed and store
    embedding=embedding_function,  # The embedding model to convert text → vectors
)

# --- STEP 6: Create a retriever from the vector store ---
# as_retriever() wraps the vector store in a LangChain Retriever interface.
# A retriever's job: given a query string, return the most relevant documents.
# Under the hood, it:
#   1. Embeds the query using the same embedding function
#   2. Performs a similarity search (cosine similarity) against stored vectors
#   3. Returns the top-k most similar chunks as Document objects
# search_kwargs={"k": 2} means "return the 2 most relevant chunks."
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# --- STEP 7: Initialize the LLM (Language Model) ---
# Create a ChatOpenAI instance using GPT-3.5 Turbo.
# This is the "generation" part of RAG — it will read the retrieved context
# and generate a natural-language answer. GPT-3.5-turbo is fast and affordable.
llm = ChatOpenAI(model="gpt-3.5-turbo")

# --- STEP 8: Define the user's question ---
# This is the query the user is asking. The RAG system will:
#   1. Find relevant chunks for this question (retrieval)
#   2. Feed those chunks + the question to the LLM (generation)
query = "What is your refund policy and how can I contact support?"

# --- STEP 9: Retrieve relevant document chunks ---
# retriever.invoke(query) does the following:
#   1. Converts the query string into a vector using OpenAIEmbeddings
#   2. Compares that query vector against all stored chunk vectors using cosine similarity
#   3. Returns the top 2 (k=2) most similar Document objects
# The result is a list of Document objects whose content is most relevant to the query.
retrieved_docs = retriever.invoke(query)

# --- STEP 10: Build the context string ---
# Join all retrieved chunks' text into a single string separated by newlines.
# This combined text becomes the "context" that the LLM will use to answer.
# For example, if 2 chunks were retrieved, this produces:
#   "Our refund policy allows returns within 30 days.\nWe offer 24/7 customer support..."
context = "\n".join([doc.page_content for doc in retrieved_docs])

# --- STEP 11: Send the prompt to the LLM and get a response ---
# llm.invoke() sends a prompt string to GPT-3.5 Turbo and returns an AIMessage object.
# The prompt uses a common RAG pattern:
#   - Instruction: "Use the context below to answer the question"
#   - Fallback: "If not found, say 'I don't know'" (prevents hallucination)
#   - Context: The retrieved chunks (grounding the LLM in real data)
#   - Question: The user's original query
# The LLM reads the context, finds the relevant info, and generates a coherent answer.
response = llm.invoke(
    f"""
Use the context below to answer the question.
If not found, say "I don't know."

Context:
{context}

Question:
{query}
"""
)

# --- STEP 12: Print the final answer ---
# response is an AIMessage object. The .content attribute holds the actual
# text string of the LLM's reply. We print it to show the user the answer.
# Expected output: A natural-language answer about the refund policy (30 days)
# and customer support (24/7 via email and chat), derived from the retrieved context.
print(response.content)
