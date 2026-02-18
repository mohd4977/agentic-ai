# Tavily is a specialized, AI-native search engine designed specifically for LLM and autonomous
# AI agents, rather than human users.
#It acts as a webaccess layer allowing AI agents to search scrape and crawl the live internet for real time,
# accurate information which helps reduce hallucination.

#Search API: Performs fast, real time web searches
#Extract API: Scrapes content from specific URLS, turning then into clean markdown
#Crawl/Map API: Allow for in-depth, site level crawling and navigation of entire websites.

#Research Endpoint (New): Enables comprehensive, end-to-end autonomous research with a single API call,
# generating detailed reports.
#Low Latency & High Speed: Optimized for sub-second, real-time results.
#Advanced Safety: Includes a built-in firewall to protect against prompt injection and data leakage.

#Reduces Hallucinations: Grounding LLM answers in real-time, verified web data.
#Developer-Friendly: Simple integration with popular frameworks like LangChain, LlamaIndex, and Agno.
#Context Optimization: Returns concise, high-signal information that reduces token usage.
#Free Tier: Offers a free plan with 1,000 monthly API credits for testing.

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

search_tool = TavilySearchResults()

agent = create_agent(model= llm,tools= [search_tool],
                     system_prompt="You are a very helpful web-searching assistant")

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "List all the jobs available in karachi for AI?"}
    ]
})

print(result["messages"][-1].content)