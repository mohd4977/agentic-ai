from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Define strict schema
class FinancialDecision(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    action: Literal["invest", "hold", "reduce"]
    reason: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm = llm.with_structured_output(FinancialDecision)

def safe_decision(prompt: str, retries=1):
    for attempt in range(retries + 1):
        try:
            result = structured_llm.invoke(prompt)

            # Business rule guardrail
            if result.risk_level == "high" and result.action == "invest":
                raise ValueError("Unsafe decision: cannot invest at high risk")

            return result

        except Exception as e:
            if attempt == retries:
                raise e
            print("Retrying due to guardrail:", str(e))


# Test input
data = """
Market volatility is high.
Revenue is decreasing.
Customer churn increased.
"""

result = safe_decision(
    f"Analyze this data and produce a financial decision:\n{data}"
)

print(result.model_dump_json(indent=2))