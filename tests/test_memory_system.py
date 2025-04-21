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

# Try importing with absolute path
try:
    from memory.memory_utils import MemoryManager
    print("Successfully imported MemoryManager")
except ImportError as e:
    print(f"Error importing MemoryManager: {e}")
    # Try alternative import approach
    sys.path.append(str(project_root / 'memory'))
    try:
        from memory import MemoryManager
        print("Successfully imported MemoryManager using alternative approach")
    except ImportError as e:
        print(f"Error importing MemoryManager using alternative approach: {e}")
        raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_system():
    """Test the memory system by adding, retrieving, and searching for memories."""

    # Initialize memory manager
    manager = MemoryManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return False

    # Add test memories
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
            "content": "User is allergic to peanuts",
            "importance": 10,
            "category": "health",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User has a dog named Max",
            "importance": 7,
            "category": "personal_info",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User prefers to read science fiction books",
            "importance": 6,
            "category": "preference",
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

    # List memories
    memories = manager.list_memories()
    logger.info(f"Retrieved {len(memories)} memories")

    # Search for memories
    search_queries = [
        "What does the user like to do outdoors?",
        "What are the user's preferences?",
        "Does the user have any pets?",
        "What should I know about the user's health?"
    ]

    for query in search_queries:
        logger.info(f"\nSearching for: {query}")
        results = manager.search_memories(query)

        if results:
            logger.info(f"Found {len(results)} relevant memories:")
            for i, memory in enumerate(results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")

    # Test memory agent integration
    try:
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.sessions import InMemorySessionService
        from memory.memory_agent import ConversationMemoryAgent

        # Load API key for OpenAI
        with open('.api_key.json', 'r') as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get('openai_api_key')

        if not openai_api_key:
            logger.error("OpenAI API key not found in .api_key.json")
            return False

        # Initialize models with cost-effective GPT-4o-mini
        storage_model = LiteLlm(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.2
        )

        retrieval_model = LiteLlm(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.3
        )

        # Try to load Gemini API key for the orchestrator
        gemini_api_key = api_keys.get('gemini_api_key')
        if gemini_api_key:
            # Use Gemini 2.0 Flash directly for the orchestrator model
            # Set the API key in the environment for direct model usage
            import os
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
            # Use the direct model name for Gemini in ADK
            orchestrator_model = "gemini-2.0-flash"  # Direct model name for Gemini in ADK
        else:
            # Fall back to GPT-4o-mini if Gemini key is not available
            orchestrator_model = LiteLlm(
                model="gpt-4o-mini",
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

        # Test processing a message
        logger.info("\nTesting memory agent with a new message")
        await memory_agent.process_message(
            message="I went hiking in the Rockies last weekend and it was amazing!",
            role="user",
            conversation_mode="chat"
        )

        # Test retrieving relevant memories
        logger.info("\nTesting memory retrieval")
        relevant_memories = await memory_agent.get_relevant_memories(
            "What outdoor activities do I enjoy?"
        )

        if relevant_memories:
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
            formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
            logger.info(formatted_memories)
        else:
            logger.info("No relevant memories retrieved")

        logger.info("Memory agent test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing memory agent: {e}")
        return False
    finally:
        # Clean up test memories
        logger.info("\nCleaning up test memories")
        for memory_id in memory_ids:
            manager.delete_memory(memory_id)
        logger.info(f"Deleted {len(memory_ids)} test memories")

if __name__ == "__main__":
    success = asyncio.run(test_memory_system())
    if success:
        logger.info("Memory system test completed successfully")
    else:
        logger.error("Memory system test failed")
