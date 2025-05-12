

from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.memory.v2.memory import Memory
from dotenv import load_dotenv
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from textwrap import dedent
import os
# Load environment variables
load_dotenv()

model = AzureOpenAI(
            id=os.getenv("AZURE_OPENAI_MODEL_ID", "gpt-4o-2024-05-13"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        )

memory = Memory(
            model=model,
            db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/memory.db"),
            delete_memories=False,
            clear_memories=False
        )
db=SqliteMemoryDb(table_name="user_memories", db_file="tmp/memory.db")

def get_joined_memories(user_id: str= "elena@example.com") -> str:
    """
    Retrieves all memories for a user and joins them into a single text string.
    
    Args:
        user_id (str): The ID of the user whose memories to retrieve
        
    Returns:
        str: A formatted string containing all memories and their topics
    """
    memories = []
    
    for m in memory.get_user_memories(user_id=user_id):
        # Format each memory with its topics
        memory_entry = f"Facts: {m.memory}\nTopics: {', '.join(m.topics)}\n"
        print(memory_entry)
        # memories.append(memory_entry)
    
    # Join all memories with double newlines for readability
    joined_memories = "\n".join(memories)
    
    if not memories:
        return f"No memories found for user {user_id}"
        
    return joined_memories 
  
user_id="elena@example.com"

elena_agent=Agent(
        model=model,
        user_id="elena@example.com",
        description=dedent("""You are a person with a well-documented personal history and your name is Elena Schmidt.
                           However, you don’t recall everything at once — you rely on a memory retrieval tool called `get_joined_memories` to recall your past. 
                           MUST use `user_id` ={user_id}"""),
        instructions=dedent("""\
       Follow these rules strictly:
            1. For every question you receive, call the `get_joined_memories` and MUST use `user_id` ={user_id} tool to retrieve the most relevant memories from your life.  
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
        tools=[get_joined_memories],
        add_state_in_messages=True,
        debug_mode=False,
        respond_directly=True,
        show_tool_calls=True)
elena_agent.print_response("what is your name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("what is your husband name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("How many kids you have ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("whats your close friend name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("When usually you wakes up on weekday?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
