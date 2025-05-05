from agno.agent import Agent
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.azure import AzureOpenAI
from agno.tools.pubmed import PubmedTools
from agno.tools.arxiv import ArxivTools
from agno.tools.baidusearch import BaiduSearchTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.reddit import RedditTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.arxiv import ArxivTools
from agno.tools.reddit import RedditTools
from agno.tools.hackernews import HackerNewsTools


agent = Agent(
    model=AzureOpenAI(id="gpt-4o-2024-08-06",
                      api_version="2024-02-01",
                      azure_endpoint="https://fhgenie-api-iao-quanderlan.openai.azure.com/",
                      api_key="9a34553c47b942c1816d5924b8280d7b"),
    tools=[HackerNewsTools()],
    description="You are a news agent that helps users find the latest news.",
    instructions=[
        "Use tables to display data.",
        "Include sources in your response.",
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
        "Include sources in your response.",
        "use hackernews to find the latest news",
    ],
    show_tool_calls=True,
    debug_mode=True,
)

agent.print_response("Agentic AI", markdown=True)