# Short-term memory:
# Stores the current conversation (chat history during the session).

# Long-term memory:
# Stores persistent data like past conversations or user preferences.

# External memory:
# Data stored outside the agent such as databases, APIs, or files.



# Import ChatOpenAI from langchain_openai package.
# This class is used to interact with OpenAI chat models (GPT models).
# It acts as the LLM (brain) of the agent.
from langchain_openai import ChatOpenAI


# Import the tool decorator.
# This is used to convert a Python function into a tool that the agent can use.
# (Not used in this script, but commonly used for agents.)
from langchain_core.tools import tool


# Import the function to create modern agents.
# This is the official LangChain 1.x method for building agents.
from langchain.agents import create_agent


# Import short-term memory class.
# This stores chat messages in RAM during the session.
from langchain_core.chat_history import InMemoryChatMessageHistory


# Import message classes.
# These represent structured chat messages.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# Import dotenv loader.
# This loads environment variables from a .env file.
from dotenv import load_dotenv 


# Load environment variables from .env
# This usually includes the OPENAI_API_KEY.
load_dotenv()


# Create the LLM (language model).
# model="gpt-4o-mini" → fast, low-cost GPT-4 level model.
# temperature=0.3 → low randomness, more logical answers.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# Create the agent.
# The agent uses the LLM and tools (none in this case).
agent = create_agent(
    model=llm, 
    tools=[],
    system_prompt="You are a helpful assistant that can remember the conversation history.",
)


# Create an in-memory chat history object.
# This will store messages during the session.
chat_history = InMemoryChatMessageHistory()


# Add a user message to memory.
# This simulates the user saying: "My name is Ali"
chat_history.add_user_message("My name is Ali")


# Send the current chat history to the agent.
# The agent will read the conversation so far.
result = agent.invoke({
    "messages": chat_history.messages
})


# Extract the agent's reply and store it in memory.
# This keeps the conversation consistent.
chat_history.add_ai_message(result["messages"][-1].content)


# Add another user message.
# The user now asks: "What is my name?"
chat_history.add_user_message("What is my name?")


# Send the updated memory to the agent again.
# Now the agent sees the full conversation.
result = agent.invoke({
    "messages": chat_history.messages
})


# Store the agent’s second reply.
chat_history.add_ai_message(result["messages"][-1].content)


# Print the entire conversation history.
# This shows all messages stored in memory.
print(chat_history.messages)

"""
1) langchain_openai

This package connects LangChain to OpenAI models.

Main purpose

Acts as the bridge between LangChain and OpenAI API.

Common modules
Chat models
from langchain_openai import ChatOpenAI


Used for:

GPT-4

GPT-4o

GPT-4.1

etc.

Embeddings
from langchain_openai import OpenAIEmbeddings


Used for:

Vector search

RAG systems

2) langchain_core

This is the foundation layer of LangChain.

It contains the core abstractions used by all integrations.

Main modules inside langchain_core
1) messages

Handles chat message objects.

from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage


Purpose:

Standard format for chat messages

Works across all LLM providers

2) tools

Used for agent tools.

from langchain_core.tools import tool


Purpose:

Convert Python functions into tools

Let agents call functions

3) chat_history

Handles memory systems.

from langchain_core.chat_history import InMemoryChatMessageHistory


Purpose:

Store conversation history

Used for short-term memory

4) prompts

Used to create prompts for LLMs.

Example:

from langchain_core.prompts import ChatPromptTemplate


Purpose:

Build structured prompts

Control LLM behavior

5) output_parsers

Used to structure LLM responses.

Example:

from langchain_core.output_parsers import StrOutputParser


Purpose:

Convert AI output into structured data

Used in pipelines

3) langchain (main package)

This is the high-level orchestration layer.

It contains:

Agents

Chains

Runnables

High-level abstractions

Important modules in langchain
1) agents
from langchain.agents import create_agent


Purpose:

Create AI agents

Manage tool usage

Control reasoning

2) chains

Used to create LLM pipelines.

Example:

from langchain.chains import LLMChain

4) dotenv

This is not a LangChain package.

It’s a Python utility for loading environment variables.

from dotenv import load_dotenv


Purpose:

Load .env file

Store API keys safely

Example .env file:

OPENAI_API_KEY=sk-xxxx
"""

"""

Short-term memory in modern LangChain

Short-term memory = current conversation storage during a session.

It’s usually implemented using chat history classes.

These are found in:

langchain_core.chat_history

Main short-term memory classes
1) InMemoryChatMessageHistory

Most basic and common.

from langchain_core.chat_history import InMemoryChatMessageHistory

Characteristics

Stored in RAM

Fast

Temporary

Lost when program stops

Use case

CLI chatbots

Testing

Prototypes

2) FileChatMessageHistory

Stores conversation in a file.

from langchain_community.chat_message_histories import FileChatMessageHistory

Characteristics

Stored on disk

Persists between runs

Simple long session memory

Use case

Local chat apps

Personal assistants

Offline agents

3) RedisChatMessageHistory

Stores memory in Redis.

from langchain_community.chat_message_histories import RedisChatMessageHistory

Characteristics

Fast in-memory database

Shared across processes

Scalable

Use case

Production chatbots

Multi-user systems

Web apps

4) PostgresChatMessageHistory

Stores memory in PostgreSQL.

from langchain_community.chat_message_histories import PostgresChatMessageHistory

Characteristics

Persistent

Structured storage

Reliable for production

Use case

SaaS apps

Customer support bots

Enterprise systems

5) MongoDBChatMessageHistory

Stores memory in MongoDB.

from langchain_community.chat_message_histories import MongoDBChatMessageHistory

Use case

NoSQL-based systems

Flexible schema apps

Where these classes live
Class	Package
InMemoryChatMessageHistory	langchain_core
FileChatMessageHistory	langchain_community
RedisChatMessageHistory	langchain_community
PostgresChatMessageHistory	langchain_community
MongoDBChatMessageHistory	langchain_community
Install community memory backends

If not installed:

pip install langchain-community

Example: File-based short-term memory
from langchain_community.chat_message_histories import FileChatMessageHistory

chat_history = FileChatMessageHistory("chat_history.json")

chat_history.add_user_message("Hello")
chat_history.add_ai_message("Hi there!")

print(chat_history.messages)


This memory persists even after restarting the script.

Quick comparison
Memory type	Stored in	Persistent
InMemory	RAM	❌ No
File	Disk file	✅ Yes
Redis	Redis DB	✅ Yes
Postgres	SQL DB	✅ Yes
MongoDB	NoSQL DB	✅ Yes
Important concept

Short-term memory is just:

Conversation storage


It doesn’t mean:

Knowledge base

Embeddings

RAG

Vector DB

Those are long-term memory.

Simple rule to remember

If you’re:

Testing → use InMemory

Building local app → use File

Building production system → use Redis/Postgres

"""