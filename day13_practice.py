from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

PERSIST_DIRECTORY = "chroma_db"
if os.path.exists(PERSIST_DIRECTORY):
    shutil.rmtree(PERSIST_DIRECTORY)
    
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    # Finance / Billing
    Document(
        page_content="Our refund policy allows returns within 30 days of purchase. "
                     "Items must be in original packaging. Digital products are non-refundable. "
                     "Refund processing takes 5-7 business days.",
        metadata={"source": "billing_docs", "department": "finance"}
    ),
    Document(
        page_content="Premium Plan costs $50/month and includes unlimited API calls, "
                     "priority support, custom integrations, and a dedicated account manager. "
                     "Annual billing saves 20%, bringing the cost to $480/year.",
        metadata={"source": "pricing_page", "department": "finance"}
    ),
    Document(
        page_content="Starter Plan is free and includes 1,000 API calls per month, "
                     "basic email support, and access to core features. "
                     "No credit card required to sign up.",
        metadata={"source": "pricing_page", "department": "finance"}
    ),
    Document(
        page_content="Enterprise Plan has custom pricing based on usage volume. "
                     "Includes 99.99% SLA, on-premise deployment, SOC2 compliance, "
                     "and a dedicated solutions architect.",
        metadata={"source": "pricing_page", "department": "finance"}
    ),
    # Support
    Document(
        page_content="We offer 24/7 customer support via email and live chat. "
                     "Premium users also get phone support. Average response time is 2 hours "
                     "for standard issues and 30 minutes for critical (P0) issues.",
        metadata={"source": "support_guide", "department": "support"}
    ),
    Document(
        page_content="To escalate an issue, email escalations@company.com or call the "
                     "support hotline at 1-800-555-0199. Critical issues (P0) are addressed "
                     "within 1 hour. Include your account ID in all communications.",
        metadata={"source": "support_guide", "department": "support"}
    ),
    # Engineering / Product
    Document(
        page_content="API rate limits: Starter plan gets 100 requests/minute, "
                     "Premium gets 1,000 requests/minute, Enterprise gets unlimited. "
                     "Rate limit headers are included in every API response.",
        metadata={"source": "api_docs", "department": "engineering"}
    ),
    Document(
        page_content="Authentication uses OAuth 2.0 with Bearer tokens. "
                     "Tokens expire after 1 hour. Use the /auth/refresh endpoint to get "
                     "a new token. API keys are generated from the dashboard settings page.",
        metadata={"source": "api_docs", "department": "engineering"}
    ),
    Document(
        page_content="New feature: AI-powered analytics dashboard released March 2026. "
                     "Available for Premium and Enterprise users. Includes predictive insights, "
                     "anomaly detection, and natural language queries for data exploration.",
        metadata={"source": "release_notes", "department": "product"}
    ),
    # Legal / Compliance
    Document(
        page_content="Data retention policy: Customer data is retained for 5 years after "
                     "last account activity. GDPR deletion requests must be fulfilled within "
                     "30 days. Users can export their data via Settings > Data Export.",
        metadata={"source": "compliance_docs", "department": "legal"}
    ),
]

test_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
test_chunks = test_splitter.split_documents(documents)

for i, chunk in enumerate(test_chunks):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n{'-'*50}")

vector_store = Chroma.from_documents(test_chunks, embedding, persist_directory=PERSIST_DIRECTORY, collection_name="company_docs")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print(f"Knowledge base ready: {vector_store._collection.count()} vectors indexed\n")

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful company assistant. Answer questions using ONLY 
the provided context. Follow these rules:
1. If the context doesn't contain the answer, say "I don't have that information."
2. Be concise but complete.
3. Cite the source document when possible.
4. If the question is ambiguous, ask for clarification."""),

    ("human", """Context:
{context}

Question: {question}"""),
])

sample_context = "Premium Plan costs $50/month with unlimited API calls."

formatted = rag_prompt.format_prompt(context=sample_context, question="How is the weather today?")

print("Formatted prompt messages:")
for msg in formatted.to_messages():
    print(f"  [{msg.type}] {msg.content}...")

def formatted_docs(docs):
    return "\n\n".join([f"{doc.page_content} (source: {doc.metadata['source']})" for doc in docs])

print(formatted_docs(documents))

rag_chain = (
    {
        "context": retriever | formatted_docs,   # retrieve docs → format as string
        "question": RunnablePassthrough(),     # pass question through as-is
    }
    | rag_prompt       # plug into prompt template
    | llm              # send to LLM
    | StrOutputParser() # extract string from AIMessage
)

print("\nQuery: 'What plans do you offer and how much do they cost?'")
result = rag_chain.invoke("What plans do you offer and how much do they cost?")
print(f"Answer: {result}")

print("\nQuery: 'How do I authenticate with the API?'")
result2 = rag_chain.invoke("How do I authenticate with the API?")
print(f"Answer: {result2}")
