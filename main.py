# Import the ChatOpenAI class from the langchain_openai module
# This class lets you interact with OpenAI chat models (like GPT-4, GPT-4o, etc.).
# It acts as a wrapper around the OpenAI API inside LangChain.
from langchain_openai import ChatOpenAI

# Imports the HumanMessage class.
# Represents a message coming from a user.
# LangChain uses structured message objects instead of raw strings.
# This helps manage multi-turn conversations.
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists).
load_dotenv()

# This is a simple example of how to use the ChatOpenAI class to interact with an OpenAI chat model.
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.9)

# We create a HumanMessage with the content "What is the meaning of life?" and pass it to the 
# llm.invoke() method.

response = llm.invoke([HumanMessage(content="What is the meaning of life?")])
print(response.content)

# What is the difference between an AI model and an AI agent?
# An AI model is a static, trained algorithm that analyzes data and generates predictions 
# or content based on inputs, acting as the "brain," 
# while an AI agent is an autonomous system that uses models to reason, 
# make decisions, and perform multi-step tasks in real-time. 

#What are the 4 components of an agent?

#AI agents are autonomous systems built on four core components: 
# a Brain (LLM) for reasoning and planning, 
# Memory to retain context and past interactions, 
# Tools to interact with external systems and take actions, and Instructions (or Perception) 
# to define goals, roles, and environmental awareness. Together, these enable the "agentic loop". 
#Here is a deeper breakdown of these four key components:
#1. The Brain (LLM & Reasoning)
#Function: Acts as the central intelligence, handling understanding, reasoning, 
# planning, and decision-making.
#Role: Processes input, breaks down complex tasks, and decides which actions to take. 
#2. Memory
#Function: Stores information, conversation history, and past experiences.
#Types: Includes short-term memory (immediate context), long-term memory (historical data), 
# and episodic memory. 
# It allows the agent to learn and maintain context over time. 
#3. Tools (Action Component)
#Function: Enables the agent to interact with the outside world, digital systems, 
# or physical environments.
#Examples: API calls, file search, code execution, browsing the web, or database queries. 
#4. Instructions (Goal & Context)
#Function: Defines the agent's purpose, behavior, boundaries, and specific tasks.
#Components: Often implemented via system prompts, goals, and utility functions 
# that guide decision-making criteria. 
#These components work in a continuous cycle: the agent receives instructions, 
# uses its brain to reason, accesses memory, uses tools to act, and 
# updates memory based on results.
# Name one agent you could build using Django.
# You could build a "Personal Finance Assistant" agent using Django. 
# This agent could help users manage their finances by tracking expenses, 
# creating budgets, and providing financial advice. 
# It could integrate with banking APIs to fetch transaction data, 
# use an LLM to analyze spending patterns, and offer personalized recommendations 
# for saving money or investing.