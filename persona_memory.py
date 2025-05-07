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
from tqdm import tqdm

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
def chunk_interview_text(text, max_tokens=2000, overlap=200):
    """
    Chunks interview text into smaller segments.
    
    Args:
        file_path (str): Path to interview text file
        max_tokens (int): Maximum tokens per chunk (approximate based on words)
        overlap (int): Number of words to overlap between chunks for context
    
    Returns:
        list: List of text chunks
    """

    # Split by interviewer questions (natural boundaries)
    segments = text.split("Interviewer: ")
    
    chunks = []
    current_chunk = ""
    word_count = 0
    
    for segment in tqdm(segments):
        if not segment.strip():
            continue
            
        # Add "Interviewer: " back except for first segment
        if segments.index(segment) > 0:
            segment = "Interviewer: " + segment
            
        # Approximate tokens by words (rough estimate)
        segment_words = len(segment.split())
        
        # If adding this segment would exceed max_tokens
        if word_count + segment_words > max_tokens:
            # Store current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous if exists
            if current_chunk:
                # Get last few sentences for overlap
                overlap_text = " ".join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + "\n\n" + segment
            else:
                current_chunk = segment
                
            word_count = segment_words
        else:
            # Add segment to current chunk
            if current_chunk:
                current_chunk += "\n\n" + segment
            else:
                current_chunk = segment
            word_count += segment_words
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# Step 2: Use LLM to extract structured memories
def extract_memories_with_llm(interview_text):
    load_dotenv()
    print(os.getenv("AZURE_OPENAI_MODEL_ID"))
    pyschologist_agent = Agent(
        model=AzureOpenAI(
            id=os.getenv("AZURE_OPENAI_MODEL_ID"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        ),
        description=dedent("""You are a psychologist that extracts structured personal facts from interview transcripts for memory storage."""),
        instructions=dedent("""
            Follow these steps:
            1. Understand the interview transcript.
            2. For each question of interviewer, extract facts and information from the interviewee answer about the persona as much as possible.
            3. Find out topics of each facts and information.
            4. if possible, extract the personality traits too in the form of facts.
            5. Collect at least 20 details about the person.
            Extract the information: Name, birthplace, early_life, family, career, relationships, health, habits, turning points, marketing, buying habits, politics, education, lifestyle, housing, children, values.
            Now below is the interview transcript:\n"""),
        response_model=FactResponse,
        use_json_mode=True
    )
    
    all_facts = []
    chunks = chunk_interview_text(interview_text)
    max_retries = 3  # Maximum number of retries per chunk
    
    for chunk in chunks:
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    output = pyschologist_agent.run(chunk)
                    # Check if output is RunResponse and has content
                    if hasattr(output, 'content') and isinstance(output.content, FactResponse):
                        facts = output.content.facts
                        all_facts.extend(facts)
                        success = True
                        print(f"✓ Successfully processed chunk with {len(facts)} facts")
                    else:
                        print(f"⚠️ Unexpected output format: {output}")
                        retry_count += 1
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print("Retrying...")
                    else:
                        print("Max retries reached, skipping chunk")
    
    if not all_facts:
        print("⚠️ No facts were successfully extracted from any chunks")
        return None
    
    try:
        return FactResponse(facts=all_facts)
    except Exception as e:
        print(f"⚠️ Error creating FactResponse: {str(e)}")
        return None
    
 
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
    transcript = load_text("interview_elena_longer.txt")
    # print(transcript)
    memories_to_remember = extract_memories_with_llm(transcript)
    # print(f"Memories to print: {memories_to_remember.content}")
    if memories_to_remember:
        store_memories(user_id="elena@example.com", memories=memories_to_remember)
