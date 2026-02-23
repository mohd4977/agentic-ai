#What is RAG?
#RAG stands for Retrieval-Augmented Generation. 
# It is a technique that combines the strengths of retrieval-based models and 
# generative models to improve the quality and relevance of generated content. 
# In RAG, a retrieval model is used to fetch relevant information from a large corpus of data, 
# which is then fed into a generative model to produce more accurate and contextually appropriate responses. 
# This approach allows the system to leverage external knowledge effectively while generating responses, 
# making it particularly useful for tasks that require up-to-date information or domain-specific knowledge.

# Instead of asking the model: "What is our refund policy?"
# 1. Retirieve relevant documentation
# 2. Inject them into prompt
# 3. Ask the model to answer using this data

# This prevents hallucination.

# RAG is used in:
# Financial systems
# Legal AI
# Healthcare AI
# internal company knowledge bots
# SaaS documentation bots

# RAG is a powerful technique for improving the accuracy and relevance of AI-generated content by combining retrieval and generation capabilities.

#RAG Architecture:
# User question -> Enbedding model -> Vector search -> Retrieve relevant documents -> 
# LLM answers using retrieved context




#OpenAIEmbeddings are specialized models that convert text into high-dimensional numerical vectors,
#capturing semantic meaning for tasks like search, clustering,
#and Retrieval-Augmented Generation (RAG).
#They enable AI to understand, compare,
#and rank information by mapping related concepts close together in a vector space.
#
#Models: Current models, such as text-embedding-3-small or
# text-embedding-3-large, offer superior performance, lower costs, and higher efficiency
# compared to older models like text-embedding-ada-002.
# Usage: They are accessed via the OpenAI API to generate embeddings for text chunks,
# which are then used for semantic searches (e.g., measuring cosine similarity).
# Integration: Commonly used within frameworks like LangChain, enabling easy implementation for RAG applications.
# Pricing: New users can receive free credits, which can be used to generate millions of tokens for embedding documents.
# Application: Ideal for applications needing to understand context,
# such as identifying similar items ("dog" vs. "puppy") or comparing large datasets.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Chroma Cloud powers serverless vector and full-text search.
# It’s extremely fast, cost-effective, scalable and painless.
# Create a DB and try it out in under 30 seconds with $5 of free credits.
# Chroma is a specialized class in the LangChain framework that provides a
# wrapper around the ChromaDB vector database.
# It allows developers to seamlessly store, embed, and retrieve text data for AI applications,
# particularly for retrieval-augmented generation (RAG).

from langchain_chroma import Chroma

# The Document class from langchain_core.documents is the fundamental data structure in
#LangChain for storing text content and associated metadata
#(e.g., source, page number) for retrieval-augmented generation (RAG).
#It is used to unify data loaded from various sources before being processed or indexed.

from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


documents = [
    Document(page_content="Our refund policy allows returns within 30 days."),
    Document(page_content="Our premium plan costs $50 per month."),
    Document(page_content="We offer 24/7 customer support via email and chat."),
    ]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

query = "How long do customers have to request a refund?"

retrieved_docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in retrieved_docs])

response = llm.invoke(
    f"""
      Use the following context to answer the question.
      If the answer is not contained within the context, say you don't know.

      Context: {context}
    Question: {query}
    """
)

print(response.content)

# How does RAG reduce hallucinations?

# RAG reduces hallucinations by grounding model responses 
# in retrieved documents. Instead of relying on pretrained knowledge, 
# the LLM answers based on dynamically retrieved context from 
# a vector database.
