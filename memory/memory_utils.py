import os
import json
import logging
import argparse
from pathlib import Path
from supabase import create_client
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.supabase_client = None
        self.nlp = None

    def load_credentials(self):
        """Load Supabase credentials from .api_key.json file."""
        try:
            api_key_path = Path('.api_key.json')
            with open(api_key_path, 'r') as f:
                api_keys = json.load(f)
                self.supabase_url = api_keys.get('supabase_url')
                self.supabase_key = api_keys.get('supabase_api_key')

            if not self.supabase_url or not self.supabase_key:
                logger.error("Supabase credentials not found or incomplete in .api_key.json")
                return False

            logger.info("Supabase credentials loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Supabase credentials: {e}")
            return False

    def connect(self):
        """Connect to Supabase and load spaCy model."""
        if not self.load_credentials():
            return False

        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase successfully")

            # Load spaCy model
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model successfully")

            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def create_embedding(self, text):
        """Create an embedding for the given text using spaCy."""
        if not self.nlp:
            logger.error("spaCy model not loaded")
            return None

        doc = self.nlp(text)
        return doc.vector.tolist()

    def list_memories(self, limit=10, user_id=None):
        """List memories from the database."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            # Build query
            query = self.supabase_client.table('memories').select('*').order('timestamp', desc=True)

            # Filter by user_id if provided
            if user_id:
                query = query.eq('user_id', user_id)

            # Execute with limit
            response = query.limit(limit).execute()

            if not response.data:
                logger.info("No memories found")
                return []

            logger.info(f"Retrieved {len(response.data)} memories")
            return response.data
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []

    def add_memory(self, content, importance, category, source="manual", conversation_mode="manual", user_id=None):
        """Add a new memory to the database.

        Args:
            content: The content of the memory
            importance: Importance rating (1-10)
            category: Category of the memory
            source: Source of the memory
            conversation_mode: Mode of conversation
            user_id: Optional user ID to associate with the memory

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            # Create embedding
            embedding = self.create_embedding(content)

            # Create memory data
            from datetime import datetime
            import time

            memory_id = str(time.time())
            memory_data = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'importance': importance,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'conversation_mode': conversation_mode,
                'user_id': user_id
            }

            # Store in Supabase
            self.supabase_client.table('memories').insert(memory_data).execute()
            logger.info(f"Memory added with ID: {memory_id}")

            # Print statement for visibility
            print(f"\n[MEMORY STORAGE] New memory added to database via memory_utils:\n  Content: {content}\n  User ID: {user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None

    def search_memories(self, query, limit=5, user_id=None):
        """Search for memories similar to the query."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return []

        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)

            # First try with the FLOAT[] version of the function
            try:
                # Search using vector similarity with proper vector format
                # The vector needs to be passed as a simple array, not as a string with ::vector cast
                # Prepare parameters
                params = {
                    'query_embedding': query_embedding,  # Pass the raw array
                    'match_threshold': 0.5,
                    'match_count': limit
                }

                # Use the appropriate function based on whether user_id is provided
                if user_id:
                    try:
                        # Try to use the user-specific function first
                        params['user_id_param'] = user_id
                        result = self.supabase_client.rpc('match_memories_by_user', params).execute()
                        logger.info(f"Using match_memories_by_user with user_id: {user_id}")
                    except Exception as e:
                        logger.error(f"Error using match_memories_by_user: {e}")
                        # Fall back to the generic function
                        del params['user_id_param']
                        result = self.supabase_client.rpc('match_memories', params).execute()
                        logger.info("Falling back to match_memories without user_id")
                else:
                    # Use the generic function
                    result = self.supabase_client.rpc('match_memories', params).execute()

                if result.data:
                    logger.info(f"Found {len(result.data)} similar memories using vector similarity")
                    return result.data
            except Exception as e:
                logger.warning(f"Error using vector similarity search: {e}")

            # If that fails, try a simple query based on category and importance
            logger.info("Falling back to basic query without vector similarity")
            query = self.supabase_client.table('memories').select('*').order('importance', desc=True)

            # Filter by user_id if provided
            if user_id:
                query = query.eq('user_id', user_id)

            result = query.limit(limit).execute()

            if not result.data:
                logger.info("No memories found")
                return []

            logger.info(f"Found {len(result.data)} memories using fallback query")

            # Add a placeholder similarity score
            for item in result.data:
                item['similarity'] = 0.5  # Default similarity score

            return result.data
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id):
        """Delete a memory from the database."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            self.supabase_client.table('memories').delete().eq('id', memory_id).execute()
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    def clear_all_memories(self):
        """Clear all memories from the database."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            self.supabase_client.rpc('clear_all_memories').execute()
            logger.info("All memories cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Memory Management Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List memories command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of memories to list")

    # Add memory command
    add_parser = subparsers.add_parser("add", help="Add a new memory")
    add_parser.add_argument("--content", required=True, help="Content of the memory")
    add_parser.add_argument("--importance", type=int, required=True, choices=range(1, 11), help="Importance score (1-10)")
    add_parser.add_argument("--category", required=True, help="Category of the memory")
    add_parser.add_argument("--source", default="manual", help="Source of the memory")

    # Search memories command
    search_parser = subparsers.add_parser("search", help="Search for memories")
    search_parser.add_argument("--query", required=True, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of memories to return")

    # Delete memory command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("--id", required=True, help="ID of the memory to delete")

    # Clear all memories command
    clear_parser = subparsers.add_parser("clear", help="Clear all memories")

    args = parser.parse_args()

    # Initialize memory manager
    manager = MemoryManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return

    # Execute command
    if args.command == "list":
        memories = manager.list_memories(args.limit)
        if memories:
            print("\nMemories:")
            for i, memory in enumerate(memories):
                print(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Importance: {memory.get('importance', 0)})")

    elif args.command == "add":
        memory_id = manager.add_memory(args.content, args.importance, args.category, args.source, user_id=None)
        if memory_id:
            print(f"Memory added successfully with ID: {memory_id}")

    elif args.command == "search":
        memories = manager.search_memories(args.query, args.limit)
        if memories:
            print("\nSearch Results:")
            for i, memory in enumerate(memories):
                similarity = memory.get('similarity', 0)
                print(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")

    elif args.command == "delete":
        success = manager.delete_memory(args.id)
        if success:
            print(f"Memory with ID {args.id} deleted successfully")

    elif args.command == "clear":
        confirm = input("Are you sure you want to clear all memories? This cannot be undone. (y/n): ")
        if confirm.lower() == 'y':
            success = manager.clear_all_memories()
            if success:
                print("All memories cleared successfully")
        else:
            print("Operation cancelled")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

