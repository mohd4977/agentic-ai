from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

PERSIST_DIR = "./chroma_db_day12"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

policy_docs = [
    Document(
        page_content="Our refund policy allows returns within 30 days of purchase. "
                     "Items must be in original packaging. Digital products are non-refundable.",
        metadata={"source": "company_policies", "department": "finance", "doc_type": "policy"}
    ),
    Document(
        page_content="Employee leave policy: Full-time employees get 20 paid leave days per year. "
                     "Sick leave is separate and unlimited with a doctor's note.",
        metadata={"source": "hr_handbook", "department": "hr", "doc_type": "policy"}
    ),
    Document(
        page_content="Data retention policy: Customer data is retained for 5 years after last activity. "
                     "GDPR deletion requests must be fulfilled within 30 days.",
        metadata={"source": "compliance_docs", "department": "legal", "doc_type": "policy"}
    ),
]

product_docs = [
    Document(
        page_content="Premium Plan: $50/month. Includes unlimited API calls, priority support, "
                     "and custom integrations. Annual billing saves 20%.",
        metadata={"source": "product_catalog", "department": "product", "doc_type": "pricing"}
    ),
    Document(
        page_content="Starter Plan: Free tier with 1,000 API calls/month. "
                     "Includes basic support via email. No SLA guarantee.",
        metadata={"source": "product_catalog", "department": "product", "doc_type": "pricing"}
    ),
    Document(
        page_content="Enterprise Plan: Custom pricing. Dedicated account manager, "
                     "99.99% SLA, on-premise deployment option, SOC2 compliance.",
        metadata={"source": "product_catalog", "department": "product", "doc_type": "pricing"}
    ),
]

support_docs = [
    Document(
        page_content="We offer 24/7 customer support via email and live chat. "
                     "Premium users get phone support. Average response time: 2 hours.",
        metadata={"source": "support_guide", "department": "support", "doc_type": "guide"}
    ),
    Document(
        page_content="To escalate an issue, email escalations@company.com or call the support hotline. "
                     "Critical issues (P0) are addressed within 1 hour.",
        metadata={"source": "support_guide", "department": "support", "doc_type": "guide"}
    ),
    Document(
        page_content="Known issue: API latency spikes between 2-4 PM UTC due to batch processing. "
                     "Workaround: schedule heavy calls outside this window.",
        metadata={"source": "support_kb", "department": "engineering", "doc_type": "known_issue"}
    ),
]

all_documents = policy_docs + product_docs + support_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])

chunks = text_splitter.split_documents(all_documents)

print(f"Sample chunk:\n{chunks[0].page_content}\nMetadata: {chunks[0].metadata}\n")
print(f"Total chunks created: {len(chunks)}")

vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR, collection_name="company_docs")

results = vector_db.similarity_search("What plans do you offer?", k=2)

print(results[0].page_content)

