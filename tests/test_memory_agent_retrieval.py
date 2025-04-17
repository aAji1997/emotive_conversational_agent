import os
import json
import logging
import asyncio
import sys
from pathlib import Path

# Get the absolute path to the project root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Print Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Import required modules
from memory.memory_utils import MemoryManager
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from memory.memory_agent import ConversationMemoryAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_agent_retrieval():
    """Test the memory agent's retrieval capabilities with various scenarios."""

    # Initialize memory manager for direct database access
    manager = MemoryManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return False

    # Load API key for OpenAI
    try:
        with open('.api_key.json', 'r') as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get('openai_api_key')

        if not openai_api_key:
            logger.error("OpenAI API key not found in .api_key.json")
            return False
    except Exception as e:
        logger.error(f"Error loading API key: {e}")
        return False

    # Initialize models
    storage_model = LiteLlm(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0.2
    )

    retrieval_model = LiteLlm(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0.3
    )

    orchestrator_model = LiteLlm(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0.1
    )

    # Initialize session service
    session_service = InMemorySessionService()

    # Initialize memory agent
    memory_agent = ConversationMemoryAgent(
        storage_agent_model=storage_model,
        retrieval_agent_model=retrieval_model,
        orchestrator_agent_model=orchestrator_model,
        session_service=session_service
    )

    # Add diverse test memories
    test_memories = [
        {
            "content": "User enjoys hiking in the mountains on weekends",
            "importance": 8,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User's favorite color is blue",
            "importance": 5,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is allergic to peanuts and has a severe reaction",
            "importance": 10,
            "category": "health",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User has a dog named Max that is a golden retriever",
            "importance": 7,
            "category": "personal_info",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User prefers to read science fiction books, especially by Isaac Asimov",
            "importance": 6,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User graduated from Stanford University with a degree in Computer Science",
            "importance": 7,
            "category": "education",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is planning a trip to Japan next summer",
            "importance": 6,
            "category": "plan",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User works as a software engineer at Google",
            "importance": 8,
            "category": "career",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User has a sister named Emma who lives in New York",
            "importance": 7,
            "category": "family",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is vegetarian and doesn't eat meat",
            "importance": 9,
            "category": "dietary",
            "source": "test",
            "conversation_mode": "chat"
        }
    ]

    # Add memories
    memory_ids = []
    for memory in test_memories:
        memory_id = manager.add_memory(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"],
            source=memory["source"],
            conversation_mode=memory["conversation_mode"]
        )
        if memory_id:
            memory_ids.append(memory_id)
            logger.info(f"Added test memory: {memory['content']}")

    if not memory_ids:
        logger.error("Failed to add any test memories")
        return False

    # Test conversation context
    logger.info("\n=== Testing memory agent with conversation context ===")

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "I'm thinking about what to do this weekend."},
        {"role": "assistant", "content": "Do you have any plans in mind?"},
        {"role": "user", "content": "I'm not sure yet. Maybe something outdoors if the weather is nice."}
    ]

    # Process the conversation
    for message in conversation:
        await memory_agent.process_message(
            message=message["content"],
            role=message["role"],
            conversation_mode="chat"
        )

    # Test retrieving relevant memories with different queries
    retrieval_queries = [
        "What outdoor activities does the user enjoy?",
        "What are the user's hobbies?",
        "What should I suggest for the user to do this weekend?",
        "What kind of food should I recommend for the user?",
        "Does the user have any pets?",
        "What is the user's profession?",
        "Where did the user go to school?",
        "What are the user's travel plans?",
        "Who is in the user's family?",
        "What are the user's reading preferences?"
    ]

    for query in retrieval_queries:
        logger.info(f"\nTesting memory retrieval for: {query}")
        relevant_memories = await memory_agent.get_relevant_memories(query)

        if relevant_memories:
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
            formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
            logger.info(formatted_memories)
        else:
            logger.info("No relevant memories retrieved")

    # Test with explicit retrieval intent
    logger.info("\n=== Testing with explicit retrieval intent ===")

    # Add a message with retrieval intent
    await memory_agent.process_message(
        message="Let me recall what outdoor activities you enjoy...",
        role="assistant",
        conversation_mode="chat",
        force_memory_retrieval=True
    )

    # Test retrieval after explicit intent
    logger.info("\nTesting memory retrieval after explicit intent")
    relevant_memories = await memory_agent.get_relevant_memories(
        "What outdoor activities does the user enjoy?",
        force_retrieval=True
    )

    if relevant_memories:
        logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
        formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
        logger.info(formatted_memories)
    else:
        logger.info("No relevant memories retrieved")

    # Test with ambiguous queries
    logger.info("\n=== Testing with ambiguous queries ===")
    ambiguous_queries = [
        "What does the user like to do?",
        "What should I know about the user?",
        "What are the user's preferences?",
        "Tell me about the user's background",
        "What's important to the user?"
    ]

    for query in ambiguous_queries:
        logger.info(f"\nTesting memory retrieval for ambiguous query: {query}")
        relevant_memories = await memory_agent.get_relevant_memories(query)

        if relevant_memories:
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
            formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
            logger.info(formatted_memories)
        else:
            logger.info("No relevant memories retrieved")

    # Test with non-existent information
    logger.info("\n=== Testing with queries about non-existent information ===")
    non_existent_queries = [
        "What musical instruments does the user play?",
        "What is the user's favorite movie?",
        "Does the user have any children?",
        "What sports does the user play?",
        "What is the user's political affiliation?"
    ]

    for query in non_existent_queries:
        logger.info(f"\nTesting memory retrieval for non-existent information: {query}")
        relevant_memories = await memory_agent.get_relevant_memories(query)

        if relevant_memories:
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
            formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
            logger.info(formatted_memories)
        else:
            logger.info("No relevant memories retrieved")

    # Clean up test memories
    logger.info("\nCleaning up test memories")
    for memory_id in memory_ids:
        manager.delete_memory(memory_id)
    logger.info(f"Deleted {len(memory_ids)} test memories")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_agent_retrieval())
    if success:
        logger.info("Memory agent retrieval test completed successfully")
    else:
        logger.error("Memory agent retrieval test failed")
