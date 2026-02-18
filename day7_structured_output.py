# How to make AI return:
# Predictable output
# Schema-based responses
# Typed outputs for backend systems

# Unstructured response:
# The revenue increased a lot, maybe around 30%, and it looks good overall.

# Structured response:
# {
#   "revenue_trend": "increase",
#   "estimated_growth": 0.30,
#   "risk_level": "low",
#   "recommendation": "continue current strategy"
# }

# pip install pydantic

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

class FinancialSummary(BaseModel):
    revenue_trend: str = Field(description="increase, decrease, or stable")
    key_insight: str
    recommendation: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm = llm.with_structured_output(FinancialSummary)

data = """
Revenue increased from 10000 in January to 15000 in February. Expenses stayed constant at 5000.
"""

result = structured_llm.invoke(
    [
        {"role": "user", "content": f"Summarize this financial data: {data}"}
    ]
)

print(result.model_dump_json(indent=2))

# How did you integrate LLM outputs into backend systems?
# I used structured outputs with schema validation, typically using Pydantic models.
# This ensured the LLM returned typed, predictable data that could be safely consumed by APIs, background jobs,
# or workflow engines.