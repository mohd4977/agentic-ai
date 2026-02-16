# Import the ChatOpenAI class which allows us to talk to OpenAI chat models
# through the LangChain interface.
from langchain_openai import ChatOpenAI

# Import the old (classic) agent utilities:
# - initialize_agent: function to create an agent
# - Tool: wrapper class that turns a Python function into an agent tool
# - AgentType: enum that defines which reasoning style the agent will use
from langchain_classic.agents import initialize_agent, Tool, AgentType

# Import dotenv loader to read environment variables from a .env file
from dotenv import load_dotenv

# Load environment variables from the .env file into the system
# This allows the OPENAI_API_KEY to be automatically available.
load_dotenv()

# Create an LLM (Large Language Model) instance
# model="gpt-4o-mini" → fast and cheap GPT-4 level model
# temperature=0.3 → low creativity, more logical answers
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define a simple calculator function
# This function will be used as a tool by the agent
def calculator_tool(input_text: str) -> str:
    """
    A simple calculator tool that evaluates basic math expressions.

    The agent will pass a string like:
    "12 * 8"

    And this function will compute and return the result.
    """
    try:
        # eval() evaluates the math expression as Python code
        # WARNING: This is unsafe for production because it can run arbitrary code.
        # It is used here only for demonstration.
        result = eval(input_text)

        # Convert the result to string because agent tools must return strings
        return str(result)

    except Exception as e:
        # If anything goes wrong, return the error message
        return f"Error: {str(e)}"

# Convert the calculator function into a Tool object
# Agents cannot use raw Python functions directly.
# They must be wrapped in a Tool class.
calculator = Tool(
    name="Calculator",              # Name of the tool (used by the agent)
    func=calculator_tool,           # The function the tool will call
    description="Useful for performing basic math."  
    # Description helps the agent decide when to use this tool
)

# Create a list of tools the agent is allowed to use
tools = [calculator]

# Initialize the agent
agent = initialize_agent(
    tools,                          # List of tools the agent can use
    llm,                            # The language model powering the agent
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # This agent type:
    # - Reads the question
    # - Decides if it needs a tool
    # - Calls the tool if necessary
    # - Returns the final answer

    verbose=True                    # Shows internal thinking steps
)

# Run the agent with a user question
# The agent will decide:
# 1. Should I use the calculator?
# 2. If yes, call the tool
# 3. Return final answer
result = agent.run("What is 12 * 8?")

# Print the final result from the agent
print(result)

ss = eval("2 + 2")
print(ss)