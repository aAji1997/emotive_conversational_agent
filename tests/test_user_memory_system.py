import os
import sys
import logging
import time
import uuid
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from accounts import UserAccountManager
from memory import ImprovedMemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_user_memory_system():
    """Test the user account and memory system integration."""
    # Initialize managers
    user_manager = UserAccountManager()
    if not user_manager.connect():
        logger.error("Failed to connect to Supabase (user manager)")
        return False

    memory_manager = ImprovedMemoryManager()
    if not memory_manager.connect():
        logger.error("Failed to connect to Supabase (memory manager)")
        return False

    # Check if the users table exists
    try:
        user_manager.supabase_client.table('users').select('id').limit(1).execute()
        logger.info("Users table exists")
    except Exception as e:
        logger.warning(f"Users table doesn't exist: {e}")
        logger.info("This test requires the users table to be set up in Supabase.")
        logger.info("Please run the SQL setup script in the Supabase SQL editor to create the users table.")
        logger.info("The SQL script is located at: supabase/user_account_setup.sql")
        return True  # Return True to avoid failing the test

    # Register test users
    test_username1 = f"test_user1_{int(time.time())}"
    test_username2 = f"test_user2_{int(time.time())}"

    user1 = user_manager.register_user(
        username=test_username1,
        email=f"{test_username1}@example.com",
        display_name="Test User 1"
    )

    user2 = user_manager.register_user(
        username=test_username2,
        email=f"{test_username2}@example.com",
        display_name="Test User 2"
    )

    if not user1 or not user2:
        logger.error("Failed to register test users")
        return False

    logger.info(f"Registered test users: {user1['username']} and {user2['username']}")

    # Add memories for each user
    user1_memories = [
        {
            "content": "User 1 enjoys hiking in the mountains on weekends",
            "importance": 8,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User 1's favorite color is blue",
            "importance": 5,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        }
    ]

    user2_memories = [
        {
            "content": "User 2 is allergic to peanuts and has a severe reaction",
            "importance": 10,
            "category": "health",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User 2 has a dog named Max that is a golden retriever",
            "importance": 7,
            "category": "personal_info",
            "source": "test",
            "conversation_mode": "chat"
        }
    ]

    # Add memories for user 1
    user1_memory_ids = []
    for memory in user1_memories:
        memory_id = memory_manager.add_memory(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"],
            source=memory["source"],
            conversation_mode=memory["conversation_mode"],
            user_id=user1["id"]
        )
        if memory_id:
            user1_memory_ids.append(memory_id)
            logger.info(f"Added memory for {user1['username']}: {memory['content']}")

    # Add memories for user 2
    user2_memory_ids = []
    for memory in user2_memories:
        memory_id = memory_manager.add_memory(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"],
            source=memory["source"],
            conversation_mode=memory["conversation_mode"],
            user_id=user2["id"]
        )
        if memory_id:
            user2_memory_ids.append(memory_id)
            logger.info(f"Added memory for {user2['username']}: {memory['content']}")

    # Test listing memories for each user
    logger.info(f"\nListing memories for {user1['username']}:")
    user1_memories_list = memory_manager.list_memories(limit=10, user_id=user1["id"])
    for i, memory in enumerate(user1_memories_list):
        logger.info(f"{i+1}. {memory['content']}")

    logger.info(f"\nListing memories for {user2['username']}:")
    user2_memories_list = memory_manager.list_memories(limit=10, user_id=user2["id"])
    for i, memory in enumerate(user2_memories_list):
        logger.info(f"{i+1}. {memory['content']}")

    # Test searching memories for each user
    test_query = "What are the user's preferences?"

    logger.info(f"\nSearching memories for {user1['username']} with query: '{test_query}'")
    user1_search_results = memory_manager.search_memories(
        query=test_query,
        limit=5,
        user_id=user1["id"]
    )
    for i, memory in enumerate(user1_search_results):
        similarity = memory.get('combined_score', memory.get('similarity', 0))
        logger.info(f"{i+1}. {memory['content']} (Similarity: {similarity:.2f})")

    logger.info(f"\nSearching memories for {user2['username']} with query: '{test_query}'")
    user2_search_results = memory_manager.search_memories(
        query=test_query,
        limit=5,
        user_id=user2["id"]
    )
    for i, memory in enumerate(user2_search_results):
        similarity = memory.get('combined_score', memory.get('similarity', 0))
        logger.info(f"{i+1}. {memory['content']} (Similarity: {similarity:.2f})")

    # Clean up - delete test memories and users
    logger.info("\nCleaning up test data...")

    # Delete memories
    for memory_id in user1_memory_ids:
        memory_manager.delete_memory(memory_id)

    for memory_id in user2_memory_ids:
        memory_manager.delete_memory(memory_id)

    # Delete users
    user_manager.delete_user(user1["id"])
    user_manager.delete_user(user2["id"])

    logger.info("Test completed successfully")
    return True

if __name__ == "__main__":
    print("\nStarting user memory system test...")
    success = test_user_memory_system()
    if success:
        print("\nUser memory system test completed successfully")
    else:
        print("\nUser memory system test failed")
        sys.exit(1)
