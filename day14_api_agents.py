# =============================================================================
# Day 14 — API-Integrated Agents: Connecting AI to External Services
# =============================================================================
# Resume lines:
#   "Integrated AI agents with APIs, internal services, and external data sources"
#   "Designed autonomous agents with context-aware prompts and real-time web search"
#
# Days 11-13: RAG (retrieval from static knowledge bases)
# TODAY — Agents that can CALL APIs and TAKE ACTIONS:
#   1. Custom tool creation (@tool decorator)
#   2. Agent + tool binding (LLM decides which tool to use)
#   3. Multi-tool agents (combining APIs)
#   4. Real API integration (live weather, web search)
#   5. Error handling for tools
#   6. Structured tool inputs (complex parameters)
# =============================================================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import requests
import json

load_dotenv()
Z  
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ════════════════════════════════════════════════════════════
# SECTION 1 — Custom Tools with @tool Decorator
# ════════════════════════════════════════════════════════════
# Tools are how agents interact with the outside world.
# The @tool decorator turns any Python function into a LangChain tool.
#
# KEY CONCEPTS:
#   - The docstring becomes the tool's description → LLM reads this
#     to decide WHEN to use the tool
#   - Type hints define the input schema → LLM knows WHAT to pass
#   - The function body does the actual work (API call, calculation, etc.)
#
# The LLM NEVER executes code — it just decides which tool to call
# and what arguments to pass. The framework executes the tool.
#
# Interview: "I created custom tools wrapping internal APIs so our
#   agent could query databases, fetch reports, and trigger workflows
#   autonomously based on user requests."

print("=" * 60)
print("SECTION 1: Custom Tools with @tool Decorator")
print("=" * 60)


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any math calculations.
    Examples: '2 + 2', '100 * 0.15', '(50 + 30) / 2'
    Only supports basic arithmetic: +, -, *, /, **, (, )"""
    # Security: only allow safe math characters
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Only basic arithmetic operations are supported."
    try:
        result = eval(expression)  # safe because we validated input above
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature, conditions,
    and wind speed. Use this when users ask about weather or temperature."""
    try:
        # Step 1: Geocode city name to coordinates (free, no API key)
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
     v  vVBG BGV bgGgb b VV vBvg HGGHAJAgh hgbqa HG gh jhq  a    geo_response = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find location: {city}"

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        location_name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")

        # Step 2: Get weather data (free Open-Meteo API, no key needed)
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_response = requests.get(weather_url, params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        }, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data["current_weather"]
        temp = current["temperature"]
        windspeed = current["windspeed"]
        # WMO weather codes mapping
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
            3: "Overcast", 45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 95: "Thunderstorm",
        }
        condition = weather_codes.get(current["weathercode"], f"Code {current['weathercode']}")

        return (f"Weather in {location_name}, {country}: {temp}°C, "
                f"{condition}, Wind: {windspeed} km/h")

    except requests.RequestException as e:
        return f"Error fetching weather for {city}: {e}"


@tool
def lookup_user(user_id: int) -> str:
    """Look up a user's profile information by their ID. Use this when someone
    asks about a specific user, user details, or user profile.
    Returns: name, email, company, and city."""
    try:
        # JSONPlaceholder — free fake API for testing
        response = requests.get(
            f"https://jsonplaceholder.typicode.com/users/{user_id}",
            timeout=10
        )
        response.raise_for_status()
        user = response.json()
        return (f"User #{user['id']}: {user['name']}\n"
                f"  Email: {user['email']}\n"
                f"  Company: {user['company']['name']}\n"
                f"  City: {user['address']['city']}")
    except requests.RequestException as e:
        return f"Error looking up user {user_id}: {e}"


@tool
def list_user_posts(user_id: int) -> str:
    """Get all blog posts written by a specific user. Use this when someone
    asks about a user's posts, articles, or content they've written."""
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/posts",
            params={"userId": user_id},
            timeout=10
        )
        response.raise_for_status()
        posts = response.json()
        if not posts:
            return f"No posts found for user {user_id}."
        result = f"Posts by user {user_id} ({len(posts)} total):\n"
        for post in posts[:5]:  # limit to 5 to keep context manageable
            result += f"  - [{post['id']}] {post['title']}\n"
        if len(posts) > 5:
            result += f"  ... and {len(posts) - 5} more posts"
        return result
    except requests.RequestException as e:
        return f"Error fetching posts for user {user_id}: {e}"


# Show tool metadata that the LLM sees
tools = [calculate, get_weather, lookup_user, list_user_posts]
print("Available tools:")
for t in tools:
    print(f"  - {t.name}: {t.description[:70]}...")
print()

# Test tools directly (before giving them to an agent)
print("Direct tool tests:")
print(f"  calculate: {calculate.invoke('(50 * 12) + 100')}")
print(f"  lookup_user: {lookup_user.invoke(1)}")


# ════════════════════════════════════════════════════════════
# SECTION 2 — ReAct Agent (Reason + Act)
# ════════════════════════════════════════════════════════════
# A ReAct agent follows this loop:
#   1. THINK — reason about what to do next
#   2. ACT — call a tool
#   3. OBSERVE — read the tool's output
#   4. REPEAT until the answer is found
#
# The LLM decides:
#   - WHICH tool to call (based on tool descriptions)
#   - WHAT arguments to pass (based on the user's question)
#   - WHEN to stop (when it has enough info to answer)
#
# create_react_agent (from langgraph) builds this loop automatically.
#
# Interview: "I used the ReAct pattern so agents could reason about
#   which API to call, inspect results, and chain multiple calls
#   to build comprehensive answers."

print("\n" + "=" * 60)
print("SECTION 2: ReAct Agent (Reason + Act)")
print("=" * 60)

# Create a ReAct agent with our tools
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful assistant with access to tools. "
           "Use the available tools to answer questions accurately. "
           "Always show your reasoning. If a tool returns an error, "
           "explain the issue to the user.",
)

# Simple single-tool query — agent should pick get_weather
print("\n--- Query: 'What is the weather in London?' ---")
result = agent.invoke({
    "messages": [HumanMessage(content="What is the weather in London right now?")]
})
print(f"Agent: {result['messages'][-1].content}")

# Math query — agent should pick calculate
print("\n--- Query: 'Calculate 15% tip on a $85 dinner' ---")
result2 = agent.invoke({
    "messages": [HumanMessage(content="Calculate 15% tip on a $85 dinner for 4 people. What's the total per person?")]
})
print(f"Agent: {result2['messages'][-1].content}")


# ════════════════════════════════════════════════════════════
# SECTION 3 — Multi-Tool Agent (Agent Chains Multiple APIs)
# ════════════════════════════════════════════════════════════
# The real power of agents: combining MULTIPLE tools in one query.
# The agent autonomously decides the sequence of API calls.
#
# Example: "Look up user 3 and show me their posts"
#   → Agent calls lookup_user(3)
#   → Agent calls list_user_posts(3)
#   → Agent combines both results into one answer
#
# This is what makes agents different from simple API wrappers —
# they ORCHESTRATE multiple services intelligently.
#
# Interview: "Our agents orchestrated calls across multiple APIs —
#   user service, content service, and analytics — combining results
#   into unified responses without manual chaining."

print("\n" + "=" * 60)
print("SECTION 3: Multi-Tool Agent (Chaining APIs)")
print("=" * 60)

# Query that requires TWO tool calls
print("\n--- Query: 'Tell me about user 3 and show their recent posts' ---")
result3 = agent.invoke({
    "messages": [HumanMessage(content="Tell me about user 3 and show their recent posts")]
})
print(f"Agent: {result3['messages'][-1].content}")

# Query that requires weather + calculation
print("\n--- Query: 'Weather in Tokyo and convert the temp to Fahrenheit' ---")
result4 = agent.invoke({
    "messages": [HumanMessage(content="What's the weather in Tokyo? Also convert the temperature to Fahrenheit.")]
})
print(f"Agent: {result4['messages'][-1].content}")


# ════════════════════════════════════════════════════════════
# SECTION 4 — Structured Tool Inputs (Complex Parameters)
# ════════════════════════════════════════════════════════════
# Real APIs often need structured inputs (multiple params, nested data).
# You can define tools with multiple parameters and the LLM will
# fill them in from the user's natural language request.
#
# This shows how agents bridge natural language → API parameters.

print("\n" + "=" * 60)
print("SECTION 4: Structured Tool Inputs")
print("=" * 60)


@tool
def search_posts(keyword: str, max_results: int = 5) -> str:
    """Search blog posts by keyword in the title. Use this when someone
    wants to find posts about a specific topic.

    Args:
        keyword: The search term to look for in post titles
        max_results: Maximum number of results to return (default: 5)
    """
    try:
        response = requests.get(
            "https://jsonplaceholder.typicode.com/posts",
            timeout=10
        )
        response.raise_for_status()
        posts = response.json()

        # Filter by keyword (case-insensitive)
        matches = [p for p in posts if keyword.lower() in p["title"].lower()]

        if not matches:
            return f"No posts found matching '{keyword}'."

        result = f"Found {len(matches)} posts matching '{keyword}'"
        result += f" (showing top {min(max_results, len(matches))}):\n"
        for post in matches[:max_results]:
            result += f"  - [{post['id']}] {post['title']} (by user {post['userId']})\n"
        return result
    except requests.RequestException as e:
        return f"Error searching posts: {e}"


@tool
def get_post_with_comments(post_id: int) -> str:
    """Get a specific blog post and its comments. Use this when someone
    wants to read a post or see the discussion on a post."""
    try:
        # Two API calls — post + comments
        post_resp = requests.get(
            f"https://jsonplaceholder.typicode.com/posts/{post_id}",
            timeout=10
        )
        post_resp.raise_for_status()
        post = post_resp.json()

        comments_resp = requests.get(
            f"https://jsonplaceholder.typicode.com/posts/{post_id}/comments",
            timeout=10
        )
        comments_resp.raise_for_status()
        comments = comments_resp.json()

        result = f"Post #{post['id']}: {post['title']}\n"
        result += f"By user {post['userId']}\n"
        result += f"Body: {post['body'][:200]}...\n\n"
        result += f"Comments ({len(comments)}):\n"
        for c in comments[:3]:
            result += f"  - {c['name']} ({c['email']}): {c['body'][:80]}...\n"
        if len(comments) > 3:
            result += f"  ... and {len(comments) - 3} more comments\n"
        return result
    except requests.RequestException as e:
        return f"Error fetching post {post_id}: {e}"


# Create new agent with expanded toolkit
expanded_tools = tools + [search_posts, get_post_with_comments]

agent_v2 = create_react_agent(
    model=llm,
    tools=expanded_tools,
    prompt="You are a helpful assistant with access to weather, math, "
           "user lookup, and blog post tools. Use the most appropriate "
           "tool(s) to answer each question. Combine information from "
           "multiple tools when needed.",
)

# Complex query requiring search → read → user lookup
print("\n--- Query: 'Find posts about \"qui\" and show me the first one with comments' ---")
result5 = agent_v2.invoke({
    "messages": [HumanMessage(content="Search for posts about 'qui' and show me the first result with its comments")]
})
print(f"Agent: {result5['messages'][-1].content}")


# ════════════════════════════════════════════════════════════
# SECTION 5 — Error Handling in Tools
# ════════════════════════════════════════════════════════════
# In production, APIs fail. Tools MUST handle errors gracefully
# and return informative error messages so the agent can:
#   - Explain the issue to the user
#   - Try an alternative approach
#   - Ask the user for different input
#
# NEVER let exceptions bubble up unhandled — the agent crashes.
# ALWAYS return a string describing what went wrong.
#
# Interview: "All our agent tools had comprehensive error handling —
#   timeouts, invalid inputs, API failures — so the agent could
#   gracefully degrade and inform users instead of crashing."

print("\n" + "=" * 60)
print("SECTION 5: Error Handling in Tools")
print("=" * 60)

# Test with invalid inputs — agent should handle gracefully
print("\n--- Query: 'Look up user 999' (non-existent user) ---")
result6 = agent_v2.invoke({
    "messages": [HumanMessage(content="Look up user 999")]
})
print(f"Agent: {result6['messages'][-1].content}")

print("\n--- Query: 'Calculate 10 / 0' (division by zero) ---")
result7 = agent_v2.invoke({
    "messages": [HumanMessage(content="What is 10 divided by 0?")]
})
print(f"Agent: {result7['messages'][-1].content}")


# ════════════════════════════════════════════════════════════
# SECTION 6 — System Prompt Engineering for API Agents
# ════════════════════════════════════════════════════════════
# The system prompt shapes HOW the agent uses tools.
# In production, you specify:
#   - When to use which tool (routing rules)
#   - How to handle ambiguity
#   - Response format expectations
#   - Guardrails on tool usage
#
# This is the "context-aware prompts" from your resume.

print("\n" + "=" * 60)
print("SECTION 6: System Prompt Engineering for API Agents")
print("=" * 60)

specialized_prompt = """You are TechCorp's customer support agent. Follow these rules:

TOOL ROUTING:
- Weather questions → use get_weather
- Math/calculations → use calculate
- User profile questions → use lookup_user
- Content/blog questions → use search_posts or list_user_posts
- Specific post details → use get_post_with_comments

BEHAVIOR:
- Always verify information using tools before answering
- If a tool returns an error, explain the issue clearly to the user
- For multi-part questions, use multiple tools and combine the results
- Be concise but helpful in your final response
- Never fabricate data — only use what the tools return"""

specialized_agent = create_react_agent(
    model=llm,
    tools=expanded_tools,
    prompt=specialized_prompt,
)

# Multi-part question testing routing
print("\n--- Complex query testing tool routing ---")
result8 = specialized_agent.invoke({
    "messages": [HumanMessage(
        content="I need three things: (1) the weather in Karachi, "
                "(2) user 5's profile, and (3) how much a 20% tip "
                "on a $120 dinner bill would be"
    )]
})
print(f"Agent: {result8['messages'][-1].content}")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DAY 14 COMPLETE — API-Integrated Agents")
print("=" * 60)
print("""
KEY TAKEAWAYS:
1. @tool decorator    → Turn any Python function into an agent tool
2. ReAct pattern      → Agent reasons → acts → observes → repeats
3. Multi-tool agents  → Autonomously chain multiple API calls
4. Structured inputs  → LLM maps natural language → API parameters
5. Error handling     → Tools must handle failures gracefully
6. System prompts     → Guide tool routing and agent behavior

TOOL ANATOMY:
  @tool
  def my_tool(param: type) -> str:
      '''Docstring = tool description (LLM reads this!)'''
      # Call API, process data, handle errors
      return "result string"

AGENT DECISION FLOW:
  User question → LLM reads tool descriptions → picks tool + args
  → Framework executes tool → LLM reads result → answers or calls
  another tool → Final answer

INTERVIEW TALKING POINTS:
- "I created custom tools wrapping internal APIs — user service,
   analytics, and CRM — so agents could fetch real-time data."
- "The ReAct pattern let agents reason about which API to call,
   inspect results, and chain calls for complex queries."
- "Every tool had error handling for timeouts, invalid inputs,
   and API failures, so agents degraded gracefully."
- "System prompts defined tool routing rules, preventing misuse
   and ensuring agents picked the right API for each query."

NEXT (Day 15): Advanced API agents — tool composition, agent
  handoff, rate limiting, and response caching.
""")
