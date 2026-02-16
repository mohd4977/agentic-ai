# Import the OpenAI chat model wrapper
from langchain_openai import ChatOpenAI

# Import the tool decorator
from langchain_core.tools import tool

# Import the modern agent creator
from langchain.agents import create_agent

# Import dotenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# Define calculator tool
@tool
def calculator(input_text: str) -> str:
    """A simple calculator tool that evaluates math expressions."""
    try:
        result = eval(input_text)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Create agent (modern API)
agent = create_agent(
    model=llm,
    tools=[calculator],
    system_prompt="You are a helpful assistant that can use tools."
)


# Run agent
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 12 * 8?"}
    ]
})

print(result["messages"][-1].content)