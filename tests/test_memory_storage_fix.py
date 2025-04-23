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

async def test_memory_storage_fix():
    """Test the memory storage fix."""
    
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
            supabase_url = api_keys.get('supabase_url')
            supabase_key = api_keys.get('supabase_api_key')
            
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
    
    # Test direct memory storage using MemoryManager
    logger.info("\n=== Test 1: Direct memory storage using MemoryManager ===")
    direct_memory = {
        "content": "User's favorite movie is The Matrix",
        "importance": 7,
        "category": "preference"
    }
    
    memory_id = manager.add_memory(
        content=direct_memory["content"],
        importance=direct_memory["importance"],
        category=direct_memory["category"],
        source="test",
        conversation_mode="test",
        user_id=test_user_id
    )
    
    if memory_id:
        logger.info(f"Successfully added direct memory with ID: {memory_id}")
    else:
        logger.error("Failed to add direct memory")
    
    # Test memory storage through memory_integration.store_memory_directly
    logger.info("\n=== Test 2: Memory storage through memory_integration.store_memory_directly ===")
    
    # Initialize memory integration with Supabase credentials
    memory_integration = MemoryIntegration(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        user_id=test_user_id,
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )
    
    integration_memory = {
        "content": "User's favorite book is Dune",
        "importance": 8,
        "category": "preference"
    }
    
    result = await memory_integration.store_memory_directly(
        content=integration_memory["content"],
        importance=integration_memory["importance"],
        category=integration_memory["category"]
    )
    
    logger.info(f"Integration direct storage result: {result}")
    
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
    success = asyncio.run(test_memory_storage_fix())
    if success:
        logger.info("Memory storage fix test completed successfully")
    else:
        logger.error("Memory storage fix test failed")
