# Import the ChatOpenAI class from langchain_openai.
# This class is a wrapper that allows LangChain to communicate
# with OpenAI’s chat-based language models (like GPT-4o).
from langchain_openai import ChatOpenAI


# Import the tool decorator.
# This decorator is used to convert a normal Python function
# into a tool that the agent can call.
from langchain_core.tools import tool


# Import the modern agent creation function.
# This is the current (LangChain 1.x) way to build agents.
from langchain.agents import create_agent


# Import the function that loads environment variables
# from a .env file.
from dotenv import load_dotenv


# Load variables from the .env file into the system.
# This is how your OPENAI_API_KEY becomes available.
load_dotenv()


# Create the LLM (Large Language Model) instance.
# This object is the "brain" of the agent.
llm = ChatOpenAI(
    model="gpt-4o-mini",   # Fast and low-cost GPT-4 level model
    temperature=0.3        # Controls randomness (lower = more logical)
)


# Define a tool using the @tool decorator.
# This tells LangChain:
# "This function can be used by the agent when needed."
@tool
def capitalize_text(input_text: str) -> str:
    """
    A simple tool that capitalizes text.

    The agent will send a string like:
    "hello world"

    And this function will return:
    "HELLO WORLD"
    """
    try:
        # Convert the input text to uppercase
        return input_text.upper()

    except Exception as e:
        # If something goes wrong, return an error message
        return f"Error: {str(e)}"


# Create the agent using the modern LangChain API.
agent = create_agent(
    model=llm,                 # The LLM that powers the agent
    tools=[capitalize_text],   # List of tools the agent can use

    # System prompt defines the agent's behavior and personality.
    # This is like instructions given to the AI.
    system_prompt="You are a helpful assistant that capitalizes text."
)


# Run the agent with a user message.
# The agent receives messages in a structured chat format.
result = agent.invoke({
    "messages": [
        {
            "role": "user",                      # Who is speaking
            "content": "Please capitalize the following text: '1'."
        }
    ]
})


# The agent returns a list of messages.
# The last message is the final response from the AI.
# We extract the text using .content
print(result["messages"][-1].content)