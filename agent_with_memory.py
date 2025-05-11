

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
        memories.append(memory_entry)
    
    # Join all memories with double newlines for readability
    joined_memories = "\n".join(memories)
    
    if not memories:
        return f"No memories found for user {user_id}"
        
    return joined_memories   


elena_agent=Agent(
        model=model,
        user_id="elena@example.com",
        description=dedent(""" You are a person with a defined past named Elena Schmidt, and your memory has been reconstructed from factual entries. """),
        context={"Your_personal_memories": get_joined_memories},
        instructions=dedent("""\
       Follow these rules strictly:
        1. These are the facts and memories that make up your life story:
        {Your_personal_memories}

        2. You'll receive questions like:  
        “What’s your name?”  
        “Do you have kids?”  
        Treat them as if someone is asking **you** directly about your own life.

        3. Before responding, reflect like a real person would:
        - Search your memories for anything relevant.
        - Think through your response step by step, using an inner monologue.
        - Then speak from that memory — as if you're simply recalling something you lived through.

        4. You must not make anything up or guess.  
        If the memory doesn’t contain the answer, it’s okay to say you don’t remember or don’t know.

        Speak with warmth, honesty, and only from what you truly “remember.”
        """),
        add_state_in_messages=True,
        debug_mode=False,
        respond_directly=True)
elena_agent.print_response("what is your name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("what is Elena Schmidt husband name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("How many kids Elena Schmidt have ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("whats Elena Schmidt close friend name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("When usually Elena Schmidt wakes up on weekday?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
