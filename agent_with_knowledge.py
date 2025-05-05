from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.azure import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.tools.reasoning import ReasoningTools
from agno.models.azure import AzureOpenAI


# Load Agno documentation in a knowledge base
knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction/agents.md"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="agno_docs",
        search_type=SearchType.hybrid,
        # Use OpenAI for embeddings
        embedder=AzureOpenAIEmbedder(id="text-embedding-ada-002-2", 
                                     dimensions=1536,
                                     api_version="2024-02-01",
                                     azure_endpoint="",
                                     api_key="s"),
    ),
)
agent = Agent(
    name="Agno Assist",
    model=AzureOpenAI(id="gpt-4o-2024-08-06",
                      api_version="2024-02-01",
                      azure_endpoint="",
                      api_key=""),
    instructions=[
        "Use tables to display data.",
        "Include sources in your response.",
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
        "include sources in your response.",
    ],
    knowledge=knowledge,
    tools=[ReasoningTools(add_instructions=True)],
    add_datetime_to_instructions=True,
    markdown=True,
)

if __name__ == "__main__":
    # Load the knowledge base, comment out after first run
    # Set recreate to True to recreate the knowledge base if needed
    agent.knowledge.load(recreate=False)
    agent.print_response(
        "What are Agents?",
        stream=True,
        show_full_reasoning=True,
        stream_intermediate_steps=True,
    )