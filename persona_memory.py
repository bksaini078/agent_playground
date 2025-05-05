from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.schema import UserMemory
from textwrap import dedent
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from dotenv import load_dotenv
from agno.models.azure import AzureOpenAI
from agno.tools.reasoning import ReasoningTools
from agno.models.lmstudio import LMStudio

load_dotenv()


# Step 1: Load interview text
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Step 2: Use LLM to extract structured memories
def extract_memories_with_llm(interview_text):
    prompt = f"""
            You are an assistant that extracts structured personal facts from interview transcripts for memory storage.

            Extract key factual information as a list of tuples in the format:
            ("fact as a short sentence", ["topic1", "topic2", ...])

            Examples of topics: early_life, family, career, relationships, health, habits, politics, education, lifestyle, housing, children, values

            Interview transcript:
            \"\"\"{interview_text}\"\"\"

            Now extract the structured memory facts:
            """
    pyschologist_agent = Agent(
    model=LMStudio(id=os.getenv("LMSTUDIO_MODEL_ID", "deepseek-r1-distill-qwen-7b")),
    tools=[],
    description=dedent("""You are an pyschologist that extracts structured personal facts from interview transcripts for memory storage."""),
    instructions=dedent("""\
        Extract key factual information as a list of tuples in the format: Nothing else
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]},
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]}
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]}
        Examples of topics: early_life, family, career, relationships, health, habits, politics, education, lifestyle, housing, children, values
        Interview transcript:
        {interview_text}"""),
    expected_output=dedent("""
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]},
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]}
        [{"fact": "short sentence of the fact", "topic":["topic1", "topic2", ...]}"""),
    reasoning=True,
    structured_outputs=True)
    output = pyschologist_agent.run().content
    print(output)
    try:
        return output.strip()
    except Exception as e:
        print("⚠️ Failed to parse response:", e)
        print("Raw output:\n", output)
        return []

# Step 3: Store extracted memories in SQLite memory DB
def store_memories(user_id, memories, db_path="tmp/memory.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    memory_db = SqliteMemoryDb(table_name="user_memories", db_file=db_path)
    memory = Memory(db=memory_db)

    for content, topics in memories:
        user_memory = UserMemory(memory=content, topics=topics)
        memory.add_user_memory(user_memory, user_id=user_id)

    print(f"Memories added for {user_id}:")
    for m in memory.get_user_memories(user_id=user_id):
        print(f"- {m.memory} (Topics: {m.topics})")

# Main
if __name__ == "__main__":
    transcript = load_text("interview_david.txt")
    memories_to_write = extract_memories_with_llm(transcript)
    print(memories_to_write)
    # if memories:
    #     store_memories(user_id="david@example.com", memories=memories)
