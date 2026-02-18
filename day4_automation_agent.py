from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

@tool
def clean_text(input_text: str) -> str:
    """
    A simple tool that removes extra whitespace from text.

    The agent will send a string like:
    "   Hello   world!  "

    And this function will return:
    "Hello world!"
    """
    try:
        # Remove leading and trailing whitespace, and reduce multiple spaces to single
        return ' '.join(input_text.split())

    except Exception as e:
        return f"Error: {str(e)}"

@tool
def summarize_text(input_text: str) -> str:
    """Summarizes the input text using the LLM."""
    response = llm.invoke([{"role": "user", "content": f"Summarize this text: {input_text}"}])
    return response.content

agent = create_agent(
    model  = llm,
    tools  = [clean_text, summarize_text],
    system_prompt = ("You are a helpful assistant that can clean and summarize text."
                     "First use the clean_text tool to remove extra whitespace, then use the summarize_text tool to create a summary of the cleaned text."
                     ),

)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Clean and summarize this text: AI is         transforming the world.",
            }
        ]
    }
)

print(result["messages"][-1].content)