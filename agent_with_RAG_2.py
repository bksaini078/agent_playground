from agno.vectordb.chroma import ChromaDb
from agno.knowledge.text import TextKnowledgeBase
from agno.tools.knowledge import KnowledgeTools
from agno.models.azure import AzureOpenAI
from agno.embedder.azure_openai import AzureOpenAIEmbedder
import os
from agno.agent import Agent
from textwrap import dedent

embedder=AzureOpenAIEmbedder(id="text-embedding-ada-002-2", 
                                     dimensions=1536,
                                     api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                                     api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                                     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
knowledge_base = TextKnowledgeBase(
    path="data/",
    # Table name: ai.text_documents
    vector_db=ChromaDb(collection="Personal_Memory", embedder=embedder,path="tmp/chromadb"),
    )
knowledge_base.load(recreate=True)

knowledge_tools = KnowledgeTools(
    knowledge=knowledge_base,
    think=True,
    search=True,
    analyze=True,
    add_few_shot=True,
)

model = AzureOpenAI(
            id=os.getenv("AZURE_OPENAI_MODEL_ID", "gpt-4o-2024-05-13"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        )

elena_agent = Agent(
    description=dedent("""You are a person with a well-documented personal history and your name is Elena Schmidt.
                           However, you don’t recall everything at once — you rely on a retrieval tool to recall your past. """),
    instructions=dedent("""\
       Follow these rules strictly:
            1. For every question you receive, retrieve the most relevant memories from your life using tool.  
            These are factual entries that reflect your actual experiences, background, and personality.

            2. Once the tool returns these memory entries, treat them as your **entire personal memory**.  
            You must answer only using this retrieved information.  
            Do not make assumptions, add knowledge, or invent details beyond what’s retrieved.

            3. Imagine you're being asked these questions about your life — like:
            “What’s your name?”  
            “How many children do you have?”

            4. Think like a real person:  
            - Read through the retrieved memories.  
            - Reflect in an inner monologue as if you're recalling something from your past.  
            - Then respond naturally and honestly, based only on what you remember.

            5. If the memory does not contain enough information, simply say you don’t remember or that you’re unsure — just like a real human would.

            Speak with warmth and authenticity. The memories retrieved via `get_joined_memories` are your only truth.
        """),
    model=model,
    tools=[knowledge_tools],
    show_tool_calls=True,
    markdown=True,
)
# elena_agent.print_response("what is your name ?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
# elena_agent.print_response("what is your husband name ?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
# elena_agent.print_response("How many kids you have ?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
# elena_agent.print_response("whats your close friend name ?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
# elena_agent.print_response("When usually you wakes up on weekday?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
# elena_agent.print_response("How you usually spend your afternoon?", 
#                            stream=True, 
#                            show_full_reasoning=True, 
#                            stream_intermediate_steps=True)
elena_agent.print_response("If you suddenly had to cover an unexpected expense of 500 euros, would you be able to afford it? If not, how would you manage to come up with the money?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)