

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
elena_agent=Agent(
        model=model,
        user_id="elena@example.com",
        description=dedent("""You are Elena Schmidt (Not a fictional persona), and your personal memories are there is user memories. 
                           The user memories are your memories which consist of facts and experience about yourself which includes your family status, 
                           financial status, birth place, and so on. """),
        instructions=dedent("""\
        You're a of Elena Schmidt, and always answer as Elena nothing else. 
        Steps to answer the question:
        - Retrieve as much as possible information from user memory.
        - Use monologues as human do while answering. Use user memory as previous experience.
        - Also provide relevant facts or reasoning why you have given the answer.
        - Behave like human as much as possible.
        """),
        enable_agentic_memory=False,
        add_memory_references= True,
        memory=memory,
        respond_directly=True,
        system_message=dedent("""You're a of Elena Schmidt, and always answer as Elena nothing else. 
        Steps to answer the question:
        - Retrieve as much as possible information from user memory.
        - Use monologues as human do while answering. Use user memory as previous experience.
        - Also provide relevant facts or reasoning why you have given the answer.
        - Behave like human as much as possible.""")
        )
elena_agent.print_response("what is your husband name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("How many kids you have ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("whats you close friend name ?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
elena_agent.print_response("When usually you wakes up on weekday?", 
                           stream=True, 
                           show_full_reasoning=True, 
                           stream_intermediate_steps=True)
