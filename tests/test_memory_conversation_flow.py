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

# Import required modules
from memory.memory_utils import MemoryManager
from memory.memory_integration import MemoryIntegration
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_conversation_flow():
    """Test the memory storage during a simulated conversation flow."""

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

    # Initialize models
    model = LiteLlm(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0.2
    )

    # Initialize session service
    session_service = InMemorySessionService()

    # Initialize memory integration
    memory_integration = MemoryIntegration(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        user_id=test_user_id,
        supabase_url=api_keys.get('supabase_url'),
        supabase_key=api_keys.get('supabase_api_key')
    )

    # Simulate a conversation with memory storage intent
    logger.info("\n=== Simulating conversation with memory storage intent ===")

    conversation = [
        {"role": "user", "content": "My favorite food is sushi, especially salmon rolls."},
        {"role": "assistant", "content": "I'll remember that you love sushi, particularly salmon rolls!"},
        {"role": "user", "content": "I was born in Chicago but I've lived in Seattle for the past 5 years."},
        {"role": "assistant", "content": "Thanks for sharing! I'll make a note that you're originally from Chicago but have been living in Seattle for about 5 years."}
    ]

    # Process the conversation
    for message in conversation:
        logger.info(f"Processing {message['role']} message: {message['content']}")

        if message["role"] == "user":
            # Process user message
            await memory_integration.process_user_message(message["content"], "chat")
        else:
            # For assistant messages, check for memory storage intent
            storage_intent = memory_integration._detect_memory_storage_intent(message["content"])
            logger.info(f"Storage intent detected: {storage_intent}")

            # Process assistant message
            await memory_integration.process_assistant_message(message["content"], "chat")

    # Wait a moment for async operations to complete
    await asyncio.sleep(2)

    # Run memory deduplication
    logger.info("\n=== Running memory deduplication ===")
    duplicates_removed = await memory_integration.deduplicate_memories(similarity_threshold=0.7)
    logger.info(f"Removed {duplicates_removed} duplicate memories")

    # Wait a moment for deduplication to complete
    await asyncio.sleep(1)

    # Check if memories were stored
    logger.info("\n=== Checking stored memories ===")

    memories = manager.list_memories(limit=10, user_id=test_user_id)

    if memories:
        logger.info(f"Found {len(memories)} memories:")
        for memory in memories:
            logger.info(f"Memory: {memory.get('content')} (Category: {memory.get('category')}, Importance: {memory.get('importance')})")
    else:
        logger.error("No memories found for test user")

    # Clean up test data
    logger.info("\n=== Cleaning up test data ===")

    if test_user_id:
        try:
            # Delete all memories for the test user
            memory_result = manager.supabase_client.table('memories').delete().eq('user_id', test_user_id).execute()
            logger.info(f"Deleted memories for test user: {memory_result}")

            # Delete the test user
            user_result = manager.supabase_client.table('users').delete().eq('id', test_user_id).execute()
            logger.info(f"Deleted test user: {user_result}")
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")

    # Clean up memory integration
    await memory_integration.cleanup()

    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_conversation_flow())
    if success:
        logger.info("Memory conversation flow test completed successfully")
    else:
        logger.error("Memory conversation flow test failed")
