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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_direct_memory_storage():
    """Test direct memory storage to diagnose issues."""
    
    # Initialize memory manager for direct database access
    manager = MemoryManager()
    
    # Connect to the database
    logger.info("Connecting to Supabase...")
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return False
    
    logger.info("Successfully connected to Supabase")
    
    # Generate a test user ID
    test_user_id = str(uuid.uuid4())
    test_username = f"test_user_{test_user_id[:8]}"
    
    # Create a test user in the database
    try:
        # Check if users table exists
        logger.info("Checking if users table exists...")
        manager.supabase_client.table('users').select('id').limit(1).execute()
        
        # Create a test user
        logger.info(f"Creating test user with ID: {test_user_id}")
        user_data = {
            "id": test_user_id,
            "username": test_username,
            "display_name": f"Test User {test_user_id[:8]}",
            "created_at": datetime.now().isoformat()
        }
        
        user_result = manager.supabase_client.table('users').insert(user_data).execute()
        logger.info(f"Created test user with ID: {test_user_id} and username: {test_username}")
        logger.info(f"User creation result: {user_result}")
    except Exception as e:
        logger.warning(f"Could not create test user: {e}")
        # If we can't create a user, use None as the user_id to avoid foreign key constraint errors
        test_user_id = None
        test_username = "anonymous"
    
    # Test direct memory storage
    logger.info("\n=== Testing direct memory storage ===")
    test_memory = {
        "content": "User's favorite color is blue",
        "importance": 7,
        "category": "preference"
    }
    
    try:
        logger.info(f"Attempting to store memory: {test_memory['content']}")
        
        # Print Supabase client details for debugging
        logger.info(f"Supabase client: {manager.supabase_client}")
        
        # Try to add the memory
        memory_id = manager.add_memory(
            content=test_memory["content"],
            importance=test_memory["importance"],
            category=test_memory["category"],
            source="test",
            conversation_mode="test",
            user_id=test_user_id
        )
        
        if memory_id:
            logger.info(f"Successfully added memory with ID: {memory_id}")
            
            # Verify the memory was stored by retrieving it
            logger.info("Verifying memory storage...")
            try:
                result = manager.supabase_client.table('memories').select('*').eq('id', memory_id).execute()
                if result.data:
                    logger.info(f"Memory verification successful: {result.data}")
                else:
                    logger.error(f"Memory verification failed: No memory found with ID {memory_id}")
            except Exception as e:
                logger.error(f"Error verifying memory: {e}")
        else:
            logger.error("Failed to add memory - no memory ID returned")
    except Exception as e:
        logger.error(f"Error during memory storage: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Clean up test data
    logger.info("\n=== Cleaning up test data ===")
    
    if test_user_id:
        try:
            # Delete all memories for the test user
            logger.info(f"Deleting memories for test user: {test_user_id}")
            memory_result = manager.supabase_client.table('memories').delete().eq('user_id', test_user_id).execute()
            logger.info(f"Memory deletion result: {memory_result}")
            
            # Delete the test user
            logger.info(f"Deleting test user: {test_user_id}")
            user_result = manager.supabase_client.table('users').delete().eq('id', test_user_id).execute()
            logger.info(f"User deletion result: {user_result}")
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")
    
    return True

if __name__ == "__main__":
    success = test_direct_memory_storage()
    if success:
        logger.info("Memory storage debug test completed")
    else:
        logger.error("Memory storage debug test failed")
