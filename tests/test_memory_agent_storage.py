import os
import json
import logging
import asyncio
import sys
import uuid
from pathlib import Path
from datetime import datetime

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
from memory.memory_integration import MemoryIntegration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_agent_storage():
    """Test the memory agent's storage capabilities with various scenarios."""

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
            gemini_api_key = api_keys.get('gemini_api_key')

        if not openai_api_key:
            logger.error("OpenAI API key not found in .api_key.json")
            return False
    except Exception as e:
        logger.error(f"Error loading API key: {e}")
        return False

    # Generate a test user ID and create a test user
    test_user_id = str(uuid.uuid4())
    test_username = f"test_user_{test_user_id[:8]}"

    # Create a test user in the database
    try:
        # Check if users table exists
        manager.supabase_client.table('users').select('id').limit(1).execute()

        # Create a test user
        user_data = {
            "id": test_user_id,
            "username": test_username,
            "display_name": f"Test User {test_user_id[:8]}",
            "created_at": datetime.now().isoformat()
        }

        manager.supabase_client.table('users').insert(user_data).execute()
        logger.info(f"Created test user with ID: {test_user_id} and username: {test_username}")
    except Exception as e:
        logger.warning(f"Could not create test user: {e}")
        # If we can't create a user, use None as the user_id to avoid foreign key constraint errors
        test_user_id = None
        test_username = "anonymous"

    logger.info(f"Testing with user ID: {test_user_id}")

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

    # Use Gemini 2.0 Flash directly for the orchestrator model
    if gemini_api_key:
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
        session_service=session_service,
        user_id=test_user_id
    )

    # Initialize memory integration
    memory_integration = MemoryIntegration(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        user_id=test_user_id,
        supabase_url=api_keys.get('supabase_url'),
        supabase_key=api_keys.get('supabase_api_key')
    )

    # Track created memories for cleanup
    created_memory_ids = []

    # Test 1: Direct memory storage
    logger.info("\n=== Test 1: Direct memory storage ===")
    direct_memories = [
        {
            "content": "User's favorite food is pizza",
            "importance": 7,
            "category": "preference"
        },
        {
            "content": "User has a cat named Whiskers",
            "importance": 6,
            "category": "personal_info"
        }
    ]

    for memory in direct_memories:
        memory_id = manager.add_memory(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"],
            source="test",
            conversation_mode="test",
            user_id=test_user_id
        )
        if memory_id:
            created_memory_ids.append(memory_id)
            logger.info(f"Added direct memory: {memory['content']} with ID: {memory_id}")
        else:
            logger.error(f"Failed to add direct memory: {memory['content']}")

    # Test 2: Memory storage through memory_integration.store_memory_directly
    logger.info("\n=== Test 2: Memory storage through memory_integration.store_memory_directly ===")
    integration_memories = [
        {
            "content": "User is learning to play the guitar",
            "importance": 5,
            "category": "hobby"
        },
        {
            "content": "User's birthday is on May 15",
            "importance": 8,
            "category": "personal_info"
        }
    ]

    for memory in integration_memories:
        result = await memory_integration.store_memory_directly(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"]
        )
        logger.info(f"Integration direct storage result: {result}")

    # Test 3: Memory storage through conversation with explicit memory intent
    logger.info("\n=== Test 3: Memory storage through conversation with explicit memory intent ===")

    # Simulate a conversation with memory storage intent
    conversation_with_intent = [
        {"role": "user", "content": "I really love chocolate ice cream, it's my favorite dessert."},
        {"role": "assistant", "content": "I'll remember that chocolate ice cream is your favorite dessert!"},
        {"role": "user", "content": "Also, I'm allergic to shellfish, so I can't eat shrimp or lobster."},
        {"role": "assistant", "content": "I'll make a note of your shellfish allergy. That's important to know."}
    ]

    # Process the conversation
    for message in conversation_with_intent:
        if message["role"] == "assistant":
            # For assistant messages, check for memory storage intent
            storage_intent = memory_integration._detect_memory_storage_intent(message["content"])
            logger.info(f"Processing assistant message with storage intent: {storage_intent}")
            await memory_integration.process_assistant_message(message["content"], "test")
        else:
            # For user messages, just process normally
            logger.info(f"Processing user message: {message['content']}")
            await memory_integration.process_user_message(message["content"], "test")

    # Test 4: Memory storage through orchestrator agent
    logger.info("\n=== Test 4: Memory storage through orchestrator agent ===")

    # Simulate a conversation with important information but no explicit memory intent
    conversation_with_info = [
        {"role": "user", "content": "I'm planning to move to Seattle next month for a new job."},
        {"role": "assistant", "content": "That sounds exciting! What kind of job will you be starting in Seattle?"},
        {"role": "user", "content": "I'll be working as a software engineer at Microsoft."},
        {"role": "assistant", "content": "Microsoft is a great company! Are you looking forward to living in Seattle?"}
    ]

    # Process the conversation
    for message in conversation_with_info:
        logger.info(f"Processing message: {message['content']}")
        await memory_agent.process_message(
            message=message["content"],
            role=message["role"],
            conversation_mode="test"
        )

    # Test 5: Verify stored memories
    logger.info("\n=== Test 5: Verify stored memories ===")

    # List all memories for the test user
    memories = manager.list_memories(limit=20, user_id=test_user_id)

    if memories:
        logger.info(f"Found {len(memories)} memories for test user:")
        for memory in memories:
            logger.info(f"Memory: {memory.get('content')} (Category: {memory.get('category')}, Importance: {memory.get('importance')})")
    else:
        logger.error("No memories found for test user")

    # Test 6: Retrieve memories to verify they can be found
    logger.info("\n=== Test 6: Retrieve memories to verify they can be found ===")

    retrieval_queries = [
        "What food does the user like?",
        "Does the user have any pets?",
        "What are the user's hobbies?",
        "What is the user's job?",
        "Where is the user moving to?",
        "Does the user have any allergies?",
        "When is the user's birthday?"
    ]

    for query in retrieval_queries:
        logger.info(f"\nTesting memory retrieval for: {query}")
        relevant_memories = await memory_agent.get_relevant_memories(query)

        if relevant_memories:
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories:")
            for memory in relevant_memories:
                logger.info(f"- {memory.get('content')} (Similarity: {memory.get('similarity', 'N/A')})")
        else:
            logger.info("No relevant memories retrieved")

    # Clean up test memories
    logger.info("\n=== Cleaning up test memories ===")

    # Delete directly created memories
    for memory_id in created_memory_ids:
        success = manager.delete_memory(memory_id)
        logger.info(f"Deleted memory {memory_id}: {success}")

    # Delete all memories for the test user
    if test_user_id:
        try:
            # Use a direct SQL query to delete all memories for the test user
            result = manager.supabase_client.table('memories').delete().eq('user_id', test_user_id).execute()
            logger.info(f"Deleted all memories for test user: {result}")

            # Delete the test user
            try:
                user_result = manager.supabase_client.table('users').delete().eq('id', test_user_id).execute()
                logger.info(f"Deleted test user: {user_result}")
            except Exception as e:
                logger.warning(f"Error deleting test user: {e}")
        except Exception as e:
            logger.error(f"Error deleting memories for test user: {e}")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_agent_storage())
    if success:
        logger.info("Memory agent storage test completed successfully")
    else:
        logger.error("Memory agent storage test failed")
