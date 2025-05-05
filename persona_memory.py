from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.schema import UserMemory
from textwrap import dedent
import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
from agno.models.azure import AzureOpenAI
from agno.models.lmstudio import LMStudio
from pydantic import BaseModel
from typing import List
from agno.models.mistral import MistralChat

load_dotenv()

class Fact(BaseModel):
    fact: str
    topic: List[str]

class FactResponse(BaseModel):
    facts: List[Fact]

# Step 1: Load interview text
def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Step 2: Use LLM to extract structured memories
def extract_memories_with_llm(interview_text):
    pyschologist_agent = Agent(
    model=AzureOpenAI(
                id=os.getenv("AZURE_MODEL_ID", "gpt-4o-2024-05-13"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            ),
    description=dedent("""You are an pyschologist that extracts structured personal facts from interview transcripts for memory storage."""),
    instructions=dedent("""
                        Follow these steps:
                        1. Understand the interview transcript.
                        2. For each question of interviewer, extract facts and information from the interviewee answer about the persona as much as possible.
                        3. Find out topics of each facts and information.
                        4. if possible, extract the personality traits too in the form of facts.
                        5. Collect atleast 20 details about the person.
                        Extract the information: Name, birthplace, early_life, family, career, relationships, health, habits, politics, education, lifestyle, housing, children, values.
                        Now below is the interview transcript:
                        Interview transcript:\n"""),
    response_model=FactResponse,
    use_json_mode=True)
    output = pyschologist_agent.run(interview_text)
    # print(output.content)
    try:
        return output
    except Exception as e:
        print("⚠️ Failed to parse response:", e)
        print("Raw output:\n", output)
        return []

# Step 3: Store extracted memories in SQLite memory DB
def store_memories(user_id, memories: FactResponse, db_path="tmp/memory.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    memory_db = SqliteMemoryDb(table_name="user_memories", db_file=db_path)
    memory = Memory(db=memory_db)

    for fact in memories.facts:
        user_memory = UserMemory(memory=fact.fact, topics=fact.topic)
        memory.add_user_memory(user_memory, user_id=user_id)

    print(f"Memories added for {user_id}:")
    for m in memory.get_user_memories(user_id=user_id):
        print(f"- {m.memory} (Topics: {m.topics})")

# Main
if __name__ == "__main__":
    transcript = load_text("interview_david.txt")
    print(transcript)
    memories_to_remember = extract_memories_with_llm(transcript)
    print(f"Memories to print: {memories_to_remember.content}")
    if memories_to_remember:
        store_memories(user_id="david@example.com", memories=memories_to_remember.content)
