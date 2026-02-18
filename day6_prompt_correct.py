# Write prompt to reduce hallucination
# Control agent behavior
# Make responses predictable
# Build production-safe agents

#Convert a weak prompt into a strong one
#add constraints and structure to outputs
#Build a reliability-focused agent


#Weak prompt
# --> You are a helpful assistant
# Problems:
# Too generic
# No constraints
# No rules
# High hallucination risk

#Strong prompt
# --> You are a financial assistant.
# --> Only answer using the provided data.
# --> If the answer is unknown, say "I dont know"
# --> Do not make up numbers or facts

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


reliable_prompt = """
You are a professional data analysis assistant.

Task:
Summarize the provided data.

Rules:
- Only use the data provided in the message.
- Do not invent facts or numbers.
- If the data is unclear, just say "insufficient data." nothing else especially if no numbers are provided.

Output format:
Summary:
Key insight:
Recommendation:
"""

unreliable_prompt = "helpful agent"

agent = create_agent(model=llm, tools=[], system_prompt=reliable_prompt)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Revenue increase"
        }
    ]
})

print(result["messages"][-1].content)

#How did you prevent hallucinations?

# I used structured system prompts with explicit constraints,
# such as restricting the model to only use provided data and
# instructing it to return “insufficient data” when information was missing.
# I also set temperature to zero for deterministic, factual responses.

# How did you control output?

# I enforced structured output formats directly in the system prompt.
# This ensured the model always returned responses in a predictable schema,
# which made it easier to integrate into backend services and validate programmatically.

# How did you ensure reliability?

# I combined prompt constraints, low temperature, structured outputs,
# and iterative testing. I tested edge cases like missing or ambiguous data and
# adjusted prompts to ensure the agent returned safe, deterministic responses instead of hallucinating.