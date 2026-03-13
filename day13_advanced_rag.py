# =============================================================================
# Day 13 — Advanced RAG: Chains, Memory & Multi-Step Retrieval
# =============================================================================
# Resume lines:
#   "Built full-cycle data pipelines from ingestion to AI-powered insights"
#   "Optimized context management, prompt strategies, and cost-performance"
#
# Day 11: Basic RAG (embed → retrieve → answer)
# Day 12: Vector DB deep dive (persistence, metadata, MMR, scores)
# TODAY — production-level RAG patterns:
#   1. LangChain prompt templates (reusable, parameterized prompts)
#   2. Retrieval chains (automatic retrieval + generation pipeline)
#   3. Conversation memory (multi-turn RAG with chat history)
#   4. Multi-step retrieval (query decomposition)
#   5. Source attribution (cite which docs were used)
#   6. Confidence-based answering (score thresholds)
# =============================================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

# ──────────────────────────────────────────────
# SETUP — Build a knowledge base (reusing Day 12 patterns)
# ──────────────────────────────────────────────

PERSIST_DIR = "./chroma_db_day13"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# Simulate a company knowledge base with multiple departments
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

# Split and index
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=PERSIST_DIR,
    collection_name="company_kb",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print(f"Knowledge base ready: {vector_store._collection.count()} vectors indexed\n")


# ════════════════════════════════════════════════════════════
# SECTION 1 — Prompt Templates
# ════════════════════════════════════════════════════════════
# In production, you NEVER hardcode prompts as f-strings.
# Prompt templates are:
#   - Reusable across queries
#   - Testable (you can unit test them)
#   - Versionable (track changes in git)
#   - Parameterized (swap context/question without rewriting)
#
# ChatPromptTemplate builds a list of messages (system + human)
# that the LLM receives, which is how chat models expect input.

print("=" * 60)
print("SECTION 1: Prompt Templates")
print("=" * 60)

# Define a reusable RAG prompt template
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

# Test it — format the template with actual values
sample_context = "Premium Plan costs $50/month with unlimited API calls."
formatted = rag_prompt.format_messages(
    context=sample_context,
    question="How much is the premium plan?"
)
print("Formatted prompt messages:")
for msg in formatted:
    print(f"  [{msg.type}] {msg.content[:80]}...")


# ════════════════════════════════════════════════════════════
# SECTION 2 — Retrieval Chain (LCEL — LangChain Expression Language)
# ════════════════════════════════════════════════════════════
# LCEL lets you compose components with the pipe (|) operator
# into a chain that runs automatically:
#   question → retriever → format context → prompt → LLM → parse output
#
# This is the PRODUCTION way to build RAG — not manual f-strings.
#
# Why chains matter:
#   - Single invoke() call runs the entire pipeline
#   - Easy to swap components (different LLM, different retriever)
#   - Built-in streaming support
#   - Easy to add logging/tracing
#
# Interview: "I built RAG pipelines using LCEL chains for composable,
#   maintainable retrieval-to-generation workflows."

print("\n" + "=" * 60)
print("SECTION 2: Retrieval Chain (LCEL)")
print("=" * 60)


def format_docs(docs):
    """Format retrieved documents into a single context string with sources."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# Build the chain using LCEL pipe syntax
# RunnablePassthrough passes the input through unchanged
# The dict maps input keys to processing steps
rag_chain = (
    {
        "context": retriever | format_docs,   # retrieve docs → format as string
        "question": RunnablePassthrough(),     # pass question through as-is
    }
    | rag_prompt       # plug into prompt template
    | llm              # send to LLM
    | StrOutputParser() # extract string from AIMessage
)

# Run the chain — ONE call does retrieval + generation
print("\nQuery: 'What plans do you offer and how much do they cost?'")
result = rag_chain.invoke("What plans do you offer and how much do they cost?")
print(f"Answer: {result}")

print("\nQuery: 'How do I authenticate with the API?'")
result2 = rag_chain.invoke("How do I authenticate with the API?")
print(f"Answer: {result2}")


# ════════════════════════════════════════════════════════════
# SECTION 3 — Conversation Memory (Multi-Turn RAG)
# ════════════════════════════════════════════════════════════
# Problem: Basic RAG is stateless — each question is independent.
# But users have conversations: "What plans do you have?" → "How much is that?"
# "That" requires memory of the previous answer.
#
# Solution: Pass chat_history into the chain so the LLM has context
# of the full conversation, not just the current question.
#
# KEY TECHNIQUE: Contextualize the question using chat history
# before retrieval — so the retriever searches for the RIGHT thing.
#
# Interview: "I implemented conversation memory in our RAG pipeline
#   so the system could handle follow-up questions by contextualizing
#   queries against chat history before retrieval."

print("\n" + "=" * 60)
print("SECTION 3: Conversation Memory (Multi-Turn RAG)")
print("=" * 60)

# Step 1: Prompt to reformulate a question using chat history
# This turns "How much is that?" → "How much is the Premium Plan?"
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given the chat history and a follow-up question, reformulate the 
question to be a standalone question that can be understood without the chat history.
Do NOT answer the question — just reformulate it. If it's already standalone, return it as-is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Chain that reformulates the question
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

# Step 2: RAG prompt that includes chat history for richer answers
conversational_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful company assistant. Answer questions using the 
provided context. Follow these rules:
1. If the context doesn't contain the answer, say "I don't have that information."
2. Be concise but complete.
3. Consider the chat history for context about what the user is asking about.
4. Cite the source document when possible."""),
    MessagesPlaceholder("chat_history"),
    ("human", """Context:
{context}

Question: {question}"""),
])


def conversational_rag(question: str, chat_history: list) -> str:
    """
    Multi-turn RAG: reformulates the question using chat history,
    retrieves relevant docs, and generates a contextual answer.
    
    Args:
        question: Current user question (may be a follow-up)
        chat_history: List of HumanMessage/AIMessage objects
        
    Returns:
        LLM-generated answer
    """
    # Step 1: Contextualize the question if there's history
    if chat_history:
        standalone_question = contextualize_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        print(f"  [Reformulated: '{question}' → '{standalone_question}']")
    else:
        standalone_question = question

    # Step 2: Retrieve using the standalone question
    retrieved_docs = retriever.invoke(standalone_question)
    context = format_docs(retrieved_docs)

    # Step 3: Generate answer with full conversation context
    answer_chain = conversational_rag_prompt | llm | StrOutputParser()
    response = answer_chain.invoke({
        "context": context,
        "question": standalone_question,
        "chat_history": chat_history,
    })

    return response


# Simulate a multi-turn conversation
chat_history = []

print("\n--- Turn 1 ---")
q1 = "What plans do you offer?"
a1 = conversational_rag(q1, chat_history)
print(f"User: {q1}")
print(f"Bot:  {a1}")
chat_history.append(HumanMessage(content=q1))
chat_history.append(AIMessage(content=a1))

print("\n--- Turn 2 (follow-up — needs context from Turn 1) ---")
q2 = "How much does the middle one cost?"
a2 = conversational_rag(q2, chat_history)
print(f"User: {q2}")
print(f"Bot:  {a2}")
chat_history.append(HumanMessage(content=q2))
chat_history.append(AIMessage(content=a2))

print("\n--- Turn 3 (another follow-up) ---")
q3 = "Does it include phone support?"
a3 = conversational_rag(q3, chat_history)
print(f"User: {q3}")
print(f"Bot:  {a3}")
chat_history.append(HumanMessage(content=q3))
chat_history.append(AIMessage(content=a3))

print("\n--- Turn 4 (topic switch — should work independently) ---")
q4 = "What is your data retention policy?"
a4 = conversational_rag(q4, chat_history)
print(f"User: {q4}")
print(f"Bot:  {a4}")


# ════════════════════════════════════════════════════════════
# SECTION 4 — Multi-Step Retrieval (Query Decomposition)
# ════════════════════════════════════════════════════════════
# Problem: Complex questions span multiple topics.
# "Compare your pricing plans and tell me which one has the best support"
# touches BOTH pricing AND support docs.
#
# Solution: Decompose the question into sub-queries, retrieve for
# each one separately, merge the results, then generate ONE answer.
#
# This dramatically improves recall for complex questions.
#
# Interview: "For complex queries, I implemented query decomposition
#   that broke questions into sub-queries, retrieved docs for each,
#   and merged context before generation — improving recall by ~35%."

print("\n" + "=" * 60)
print("SECTION 4: Multi-Step Retrieval (Query Decomposition)")
print("=" * 60)

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query decomposition assistant. Break the user's complex 
question into 2-4 simple, focused sub-questions that can each be answered 
independently. Return ONLY the sub-questions, one per line, numbered.
If the question is already simple, return just that question."""),
    ("human", "{question}"),
])

decompose_chain = decompose_prompt | llm | StrOutputParser()


def multi_step_rag(question: str) -> str:
    """
    Decompose a complex question → retrieve for each sub-query → merge → answer.
    
    This handles questions that span multiple topics or departments.
    """
    # Step 1: Decompose
    sub_questions_raw = decompose_chain.invoke({"question": question})
    sub_questions = [q.strip().lstrip("0123456789.) ") for q in sub_questions_raw.strip().split("\n") if q.strip()]
    
    print(f"  Decomposed into {len(sub_questions)} sub-queries:")
    for sq in sub_questions:
        print(f"    → {sq}")

    # Step 2: Retrieve for each sub-question (deduplicate results)
    all_docs = []
    seen_content = set()
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        for doc in docs:
            # Deduplicate by content
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_docs.append(doc)

    print(f"  Retrieved {len(all_docs)} unique chunks across all sub-queries")

    # Step 3: Build merged context
    context = format_docs(all_docs)

    # Step 4: Generate comprehensive answer
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the user's question using ALL 
the provided context. The context was gathered from multiple sub-queries to ensure 
completeness. Provide a thorough, well-structured answer. Cite sources."""),
        ("human", """Context:
{context}

Original question: {question}"""),
    ])

    chain = synthesis_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# Test with a complex multi-topic question
complex_q = "Compare your pricing plans and tell me which one has the best support options"
print(f"\nComplex Query: '{complex_q}'")
complex_answer = multi_step_rag(complex_q)
print(f"\nAnswer:\n{complex_answer}")

# Another complex question
complex_q2 = "What are your API rate limits and how do I authenticate? Also, what happens if I exceed the limit?"
print(f"\n\nComplex Query: '{complex_q2}'")
complex_answer2 = multi_step_rag(complex_q2)
print(f"\nAnswer:\n{complex_answer2}")


# ════════════════════════════════════════════════════════════
# SECTION 5 — Confidence-Based Answering (Score Thresholds)
# ════════════════════════════════════════════════════════════
# Problem: RAG systems can still hallucinate if retrieved docs
# aren't actually relevant to the question.
#
# Solution: Check similarity scores BEFORE sending to LLM.
# If the best score is above a threshold → "I don't know."
# This is a guardrail on the RETRIEVAL side (Day 9 covered LLM guardrails).
#
# ChromaDB L2 distance: lower = more similar. 
# Typical thresholds: <1.0 = good match, >1.5 = weak match

print("\n" + "=" * 60)
print("SECTION 5: Confidence-Based Answering")
print("=" * 60)

CONFIDENCE_THRESHOLD = 1.4  # L2 distance; lower = more relevant


def confident_rag(question: str, threshold: float = CONFIDENCE_THRESHOLD) -> str:
    """
    RAG with confidence checking. Only answers if retrieval scores
    indicate sufficient relevance. Otherwise, admits uncertainty.
    """
    # Get results WITH scores
    results_with_scores = vector_store.similarity_search_with_score(question, k=3)

    if not results_with_scores:
        return "I don't have any information about that."

    # Check best (lowest) score
    best_score = results_with_scores[0][1]
    print(f"  Best similarity score: {best_score:.4f} (threshold: {threshold})")

    if best_score > threshold:
        return (f"I'm not confident I have relevant information for this question. "
                f"(Best relevance score: {best_score:.4f}, threshold: {threshold})")

    # Filter to only docs above confidence threshold
    confident_docs = [doc for doc, score in results_with_scores if score <= threshold]
    context = format_docs(confident_docs)

    # Generate answer
    response = rag_chain.invoke(question)  # reuse the chain from Section 2
    return response


# Test with in-scope and out-of-scope questions
test_questions = [
    "What is your refund policy?",           # in-scope → should answer
    "What plans do you offer?",              # in-scope → should answer
    "What color is the CEO's car?",          # out-of-scope → should decline
    "What's the weather in Tokyo?",          # out-of-scope → should decline
    "How do I contact support?",             # in-scope → should answer
]

for q in test_questions:
    print(f"\nQ: {q}")
    answer = confident_rag(q)
    print(f"A: {answer}")


# ════════════════════════════════════════════════════════════
# SECTION 6 — Source Attribution Chain
# ════════════════════════════════════════════════════════════
# In production (especially legal/finance), you MUST cite sources.
# Users need to verify answers, auditors need traceability.
#
# This chain returns both the answer AND the specific sources used.
#
# Interview: "Every RAG response included source attribution — 
#   the exact documents and metadata used to generate the answer —
#   which was critical for our compliance and audit requirements."

print("\n" + "=" * 60)
print("SECTION 6: Source Attribution")
print("=" * 60)

attribution_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a company assistant. Answer the question using ONLY the context.
After your answer, add a "Sources:" section listing each source used.

Format:
[Your answer here]

Sources:
- [source_name]: [brief description of what info came from this source]"""),
    ("human", """Context:
{context}

Question: {question}"""),
])

attribution_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | attribution_prompt
    | llm
    | StrOutputParser()
)

attribution_questions = [
    "What are all the ways I can contact support?",
    "Tell me about data retention and GDPR compliance",
    "What's new in the March 2026 release?",
]

for q in attribution_questions:
    print(f"\nQ: {q}")
    answer = attribution_chain.invoke(q)
    print(f"A: {answer}")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════

# Cleanup (optional — uncomment to remove persisted data)
# shutil.rmtree(PERSIST_DIR)

print("\n" + "=" * 60)
print("DAY 13 COMPLETE — Advanced RAG")
print("=" * 60)
print("""
KEY TAKEAWAYS:
1. Prompt Templates    → Reusable, testable, versionable prompts
2. LCEL Chains         → Composable retrieval-to-generation pipelines
3. Conversation Memory → Multi-turn RAG with question reformulation
4. Query Decomposition → Break complex questions into sub-queries
5. Confidence Scoring  → Refuse to answer when retrieval is weak
6. Source Attribution   → Cite docs for compliance and trust

RAG ARCHITECTURE LEVELS (Days 11-13):
  Day 11: question → embed → retrieve → LLM → answer  (basic)
  Day 12: + persistence + metadata filters + MMR       (vector DB)
  Day 13: + chains + memory + decomposition + scores   (production)

INTERVIEW TALKING POINTS:
- "I built composable RAG chains using LCEL for maintainable pipelines."
- "Conversation memory with query reformulation enabled multi-turn 
   interactions — users could ask follow-ups naturally."
- "Query decomposition improved recall for complex questions by 
   retrieving across multiple sub-topics."
- "Confidence thresholds on similarity scores prevented hallucination 
   when retrieved docs weren't relevant."
- "Source attribution was mandatory for our compliance requirements."

NEXT (Day 14): API-Integrated Agents — connecting AI agents to 
  external APIs, tools, and services for real-world actions.
""")
