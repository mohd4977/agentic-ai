# =============================================================================
# Day 12 — Deep Dive: Vector Database Usage with ChromaDB
# =============================================================================
# Resume line: "Built RAG pipelines using embeddings and vector databases
#               for accurate knowledge retrieval."
#
# Day 11 & Day 12 practice: basic in-memory RAG
# TODAY: production-grade vector DB skills:
#   1. Persistent vector store (save/load from disk)
#   2. Metadata filtering (targeted retrieval)
#   3. Multi-source document ingestion
#   4. Similarity search WITH scores (relevance ranking)
#   5. Collection management (CRUD operations)
#   6. MMR (Maximal Marginal Relevance) for diverse results
# =============================================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# ──────────────────────────────────────────────
# SECTION 1 — Persistent Vector Store
# ──────────────────────────────────────────────
# In production, you NEVER want an in-memory vector store.
# Every restart would re-embed all documents (slow + expensive).
# Solution: persist to disk. ChromaDB supports this natively.
#
# KEY CONCEPTS:
#   persist_directory → folder where vectors are saved on disk
#   collection_name   → logical grouping (like a DB table)
#   On next load, Chroma reads from disk — no re-embedding needed.

print("=" * 60)
print("SECTION 1: Persistent Vector Store")
print("=" * 60)

PERSIST_DIR = "./chroma_db_day12"

# Clean up from previous runs so we start fresh for demo
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# ──────────────────────────────────────────────
# SECTION 2 — Multi-Source Document Ingestion with Metadata
# ──────────────────────────────────────────────
# In real systems, documents come from MULTIPLE sources:
#   - Company policies, product docs, support tickets, etc.
# Metadata lets you:
#   1. Track WHERE each chunk came from
#   2. FILTER results at query time (e.g., "only search policies")
#   3. Add timestamps, versions, departments, etc.
#
# This is how you'd answer in an interview:
#   "We tagged every document with source, department, and date
#    metadata so agents could do targeted retrieval instead of
#    searching the entire corpus."

print("\n" + "=" * 60)
print("SECTION 2: Multi-Source Ingestion with Metadata")
print("=" * 60)

# Simulate documents from different company departments
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

# Combine all documents
all_documents = policy_docs + product_docs + support_docs

# ──────────────────────────────────────────────
# SECTION 3 — Text Splitting with Metadata Preservation
# ──────────────────────────────────────────────
# When you split documents, metadata is automatically copied
# to each chunk. This is critical — without it, you lose
# the ability to filter by source after splitting.
#
# chunk_size=200  → bigger chunks preserve more context
# chunk_overlap=30 → overlap prevents losing boundary info

print("\n" + "=" * 60)
print("SECTION 3: Splitting with Metadata Preservation")
print("=" * 60)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""]  # tries these in order
)

chunks = text_splitter.split_documents(all_documents)

print(f"Original documents: {len(all_documents)}")
print(f"After splitting:    {len(chunks)} chunks")
print(f"\nSample chunk metadata: {chunks[0].metadata}")
print(f"Sample chunk content:  {chunks[0].page_content[:80]}...")

# ──────────────────────────────────────────────
# SECTION 4 — Create Persistent Vector Store with Collection
# ──────────────────────────────────────────────
# collection_name groups vectors logically.
# In production you might have:
#   - "company_knowledge" for internal docs
#   - "customer_tickets" for support history
#   - "product_docs" for technical documentation

print("\n" + "=" * 60)
print("SECTION 4: Persistent Vector Store Creation")
print("=" * 60)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=PERSIST_DIR,         # saves to disk automatically
    collection_name="company_knowledge",   # logical grouping
)

print(f"Vector store created at: {PERSIST_DIR}")
print(f"Collection: 'company_knowledge'")
print(f"Total vectors stored: {vector_store._collection.count()}")

# ──────────────────────────────────────────────
# SECTION 5 — Metadata Filtering (Targeted Retrieval)
# ──────────────────────────────────────────────
# This is a KEY production skill. Instead of searching ALL vectors,
# you filter by metadata FIRST, then do similarity search within
# the filtered subset. This is:
#   - Faster (smaller search space)
#   - More accurate (no irrelevant results from wrong departments)
#   - Cost-effective (fewer tokens sent to LLM)
#
# Interview answer: "We used metadata filters to scope retrieval
#   to specific departments, reducing noise and improving accuracy
#   by 40% compared to unfiltered search."

print("\n" + "=" * 60)
print("SECTION 5: Metadata Filtering")
print("=" * 60)

# 5a. Unfiltered search — searches everything
print("\n--- 5a. Unfiltered search: 'What plans do you offer?' ---")
results_all = vector_store.similarity_search("What plans do you offer?", k=3)
for i, doc in enumerate(results_all):
    print(f"  [{i+1}] ({doc.metadata.get('department', '?')}) {doc.page_content[:80]}...")

# 5b. Filtered search — only product department
print("\n--- 5b. Filtered (department=product): 'What plans do you offer?' ---")
results_filtered = vector_store.similarity_search(
    "What plans do you offer?",
    k=3,
    filter={"department": "product"}  # only search product docs
)
for i, doc in enumerate(results_filtered):
    print(f"  [{i+1}] ({doc.metadata.get('department', '?')}) {doc.page_content[:80]}...")

# 5c. Filter by doc_type
print("\n--- 5c. Filtered (doc_type=policy): 'data deletion' ---")
results_policy = vector_store.similarity_search(
    "data deletion",
    k=2,
    filter={"doc_type": "policy"}
)
for i, doc in enumerate(results_policy):
    print(f"  [{i+1}] ({doc.metadata.get('source', '?')}) {doc.page_content[:80]}...")

# ──────────────────────────────────────────────
# SECTION 6 — Similarity Search WITH Scores
# ──────────────────────────────────────────────
# In production, you need to know HOW relevant each result is.
# similarity_search_with_score() returns (Document, score) tuples.
#
# ChromaDB uses L2 (Euclidean) distance by default:
#   Lower score = MORE similar (0 = perfect match)
#
# Use case: Set a threshold — if best score > 0.5, tell the user
# "I don't have enough information" instead of hallucinating.

print("\n" + "=" * 60)
print("SECTION 6: Similarity Search with Scores")
print("=" * 60)

queries = [
    "How do I get a refund?",               # should match well
    "What's the weather like on Mars?",      # should NOT match well
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results_with_scores = vector_store.similarity_search_with_score(query, k=2)
    for doc, score in results_with_scores:
        print(f"  Score: {score:.4f} | {doc.page_content[:70]}...")

# ──────────────────────────────────────────────
# SECTION 7 — MMR: Maximal Marginal Relevance
# ──────────────────────────────────────────────
# Problem: Standard similarity search can return redundant results
#   (3 chunks all saying nearly the same thing).
#
# Solution: MMR balances RELEVANCE and DIVERSITY.
#   - fetch_k: initially retrieve this many candidates
#   - k: final number to return after diversity re-ranking
#   - lambda_mult: 0 = max diversity, 1 = max relevance
#
# Interview answer: "We used MMR retrieval to ensure the LLM
#   received diverse context, reducing redundant information
#   and improving answer completeness."

print("\n" + "=" * 60)
print("SECTION 7: MMR (Maximal Marginal Relevance)")
print("=" * 60)

# Standard retriever — may return similar/redundant chunks
retriever_standard = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# MMR retriever — balances relevance + diversity
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # return 3 results
        "fetch_k": 10,   # consider top 10 candidates first
        "lambda_mult": 0.5  # balance between relevance and diversity
    }
)

query = "Tell me about pricing and support"

print(f"\nQuery: '{query}'")
print("\n--- Standard Retrieval ---")
standard_results = retriever_standard.invoke(query)
for i, doc in enumerate(standard_results):
    print(f"  [{i+1}] ({doc.metadata.get('department', '?')}) {doc.page_content[:80]}...")

print("\n--- MMR Retrieval (more diverse) ---")
mmr_results = retriever_mmr.invoke(query)
for i, doc in enumerate(mmr_results):
    print(f"  [{i+1}] ({doc.metadata.get('department', '?')}) {doc.page_content[:80]}...")

# ──────────────────────────────────────────────
# SECTION 8 — Loading a Persisted Vector Store
# ──────────────────────────────────────────────
# This is the PAYOFF of persistence.
# In production, you embed documents ONCE (during ingestion),
# then load the existing vector store on every request.
# No re-embedding = fast startup + zero embedding cost.

print("\n" + "=" * 60)
print("SECTION 8: Loading Persisted Vector Store")
print("=" * 60)

# Simulate a "new session" by loading from disk
loaded_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_function,
    collection_name="company_knowledge",
)

print(f"Loaded {loaded_store._collection.count()} vectors from disk")

# Verify it works
test_results = loaded_store.similarity_search("refund policy", k=1)
print(f"Test query result: {test_results[0].page_content[:80]}...")

# ──────────────────────────────────────────────
# SECTION 9 — Adding Documents to Existing Store
# ──────────────────────────────────────────────
# In production, knowledge bases grow. You add NEW documents
# to existing vector stores without re-embedding everything.
# This is incremental ingestion.

print("\n" + "=" * 60)
print("SECTION 9: Incremental Document Addition")
print("=" * 60)

count_before = loaded_store._collection.count()

new_docs = [
    Document(
        page_content="New feature released: AI-powered analytics dashboard. "
                     "Available for Premium and Enterprise users starting March 2026.",
        metadata={"source": "release_notes", "department": "product", "doc_type": "announcement"}
    ),
    Document(
        page_content="Security update: All API keys must be rotated by April 2026. "
                     "Use the dashboard settings page to generate new keys.",
        metadata={"source": "security_bulletin", "department": "engineering", "doc_type": "announcement"}
    ),
]

# Split and add
new_chunks = text_splitter.split_documents(new_docs)
loaded_store.add_documents(new_chunks)

count_after = loaded_store._collection.count()
print(f"Vectors before: {count_before}")
print(f"Vectors after:  {count_after} (+{count_after - count_before} new)")

# Verify new content is searchable
new_results = loaded_store.similarity_search("AI analytics dashboard", k=1)
print(f"New doc found: {new_results[0].page_content[:80]}...")

# ──────────────────────────────────────────────
# SECTION 10 — Full RAG Pipeline with Filtered Retrieval
# ──────────────────────────────────────────────
# Putting it all together: metadata-filtered retrieval + LLM generation.
# This is what a production RAG system looks like.

print("\n" + "=" * 60)
print("SECTION 10: Full RAG Pipeline with Filtering")
print("=" * 60)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def rag_query(question: str, department_filter: str = None) -> str:
    """
    Production-style RAG function with optional department filtering.

    Args:
        question: The user's question
        department_filter: Optional department to scope the search

    Returns:
        LLM-generated answer grounded in retrieved context
    """
    # Step 1: Retrieve with optional filter
    search_kwargs = {"k": 3}
    if department_filter:
        search_kwargs["filter"] = {"department": department_filter}

    results = loaded_store.similarity_search(question, **search_kwargs)

    # Step 2: Check relevance (use scores in production)
    if not results:
        return "I don't have information about that topic."

    # Step 3: Build context with source attribution
    context_parts = []
    sources = set()
    for doc in results:
        context_parts.append(doc.page_content)
        sources.add(doc.metadata.get("source", "unknown"))

    context = "\n---\n".join(context_parts)

    # Step 4: Generate answer with structured prompt
    prompt = f"""You are a helpful company assistant. Answer the question using ONLY
the provided context. If the context doesn't contain the answer, say
"I don't have that information."

Include which source the information came from when possible.

Context:
{context}

Sources available: {', '.join(sources)}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


# Test the full pipeline
test_questions = [
    ("What is your refund policy?", None),
    ("What plans do you offer?", "product"),
    ("How do I escalate a support issue?", "support"),
    ("What are the data retention rules?", "legal"),
    ("What new features were released?", None),
]

for question, dept in test_questions:
    filter_text = f" [filter: {dept}]" if dept else " [no filter]"
    print(f"\nQ: {question}{filter_text}")
    answer = rag_query(question, dept)
    print(f"A: {answer}")

# ──────────────────────────────────────────────
# SECTION 11 — Cleanup (optional for demo)
# ──────────────────────────────────────────────
# In production you'd never delete — this is just for clean demo runs.
# Uncomment below to clean up the persisted DB:
# shutil.rmtree(PERSIST_DIR)
# print("\nCleaned up persisted vector store.")

print("\n" + "=" * 60)
print("DAY 12 COMPLETE — Vector DB Deep Dive")
print("=" * 60)
print("""
KEY TAKEAWAYS:
1. Always use persistent storage in production (persist_directory)
2. Tag documents with rich metadata for filtered retrieval
3. Use similarity scores to detect low-confidence results
4. MMR retrieval reduces redundancy in context
5. Incremental ingestion lets you grow the knowledge base
6. Metadata filtering = faster, cheaper, more accurate retrieval

INTERVIEW TALKING POINTS:
- "I built RAG pipelines with ChromaDB, using metadata-filtered 
   retrieval to scope searches by department and doc type."
- "We used persistent vector stores to avoid re-embedding costs 
   on every deployment."
- "MMR retrieval ensured diverse context, improving answer quality."
- "Similarity scores let us implement confidence thresholds to 
   prevent hallucination on out-of-scope questions."

NEXT (Day 13): Advanced RAG — chains, prompt templates, 
  conversation memory, and multi-step retrieval.
""")
