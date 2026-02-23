# Handle tool failures
# Recover from bad LLM outputs
# Retry failed agent steps
# Make agents production safe


# tool crash - API timeout
# Bad LLM output - Invalid JSON
# Network error - Search API down
# Rate limit Too many requests

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

@tool
def divide_numbers(input_text: str) -> str:
    """Safely divide numbers and handle errors like division by zero."""
    try:
        result = eval(input_text)
        return str(result)
    except Exception as e:
        return f"Tool error: {str(e)}"

agent = create_agent(
    model=llm,
    tools=[divide_numbers],
    system_prompt="You are a helpful assistant that can perform calculations. If a tool returns an error, explain it to user clearly and suggest a fix.",
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Calculate 10 / 0"
        }
    ]
})

print(result["messages"][-1].content)

# How did you handle failures in your agent systems?

# I implemented error handling at both the tool and agent levels.
# Tools returned structured error messages instead of raising exceptions,
# and the agent layer included retry logic for transient failures like API timeouts or schema validation errors.
# This made the system more resilient and production-safe.
