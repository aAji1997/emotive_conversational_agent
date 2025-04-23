import os
import json
import logging
import argparse
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pathlib import Path
import spacy
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.nlp = None

    def load_env_variables(self):
        """Load environment variables from .env file."""
        try:
            load_dotenv()
            self.user = os.getenv("user")
            self.password = os.getenv("password")
            self.host = os.getenv("host")
            self.port = os.getenv("port")
            self.dbname = os.getenv("dbname")

            if not all([self.user, self.password, self.host, self.port, self.dbname]):
                logger.error("Missing required environment variables in .env file")
                return False

            logger.info("Environment variables loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading environment variables: {e}")
            return False

    def connect(self):
        """Connect to the PostgreSQL database and load spaCy model."""
        if not self.load_env_variables():
            return False

        try:
            # Connect to the database
            self.connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.dbname
            )
            # Use RealDictCursor to get results as dictionaries
            self.cursor = self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            logger.info("Connected to the database successfully")

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
        if not self.cursor:
            logger.error("Not connected to the database")
            return []

        try:
            # Build query
            query = "SELECT * FROM memories"
            params = []

            # Add user_id filter if provided
            if user_id:
                query += " WHERE user_id = %s"
                params.append(user_id)

            # Add order and limit
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            self.cursor.execute(query, tuple(params))
            memories = self.cursor.fetchall()

            if not memories:
                logger.info("No memories found")
                return []

            logger.info(f"Retrieved {len(memories)} memories")
            return memories
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
        if not self.cursor:
            logger.error("Not connected to the database")
            return None

        try:
            # Create embedding
            embedding = self.create_embedding(content)

            # Create memory data
            memory_id = str(time.time())
            timestamp = datetime.now().isoformat()

            # Insert the memory
            self.cursor.execute(
                """
                INSERT INTO memories
                (id, content, embedding, importance, category, timestamp, source, conversation_mode, user_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    memory_id,
                    content,
                    embedding,
                    importance,
                    category,
                    timestamp,
                    source,
                    conversation_mode,
                    user_id
                )
            )
            self.connection.commit()

            logger.info(f"Memory added with ID: {memory_id}")

            # Print statement for visibility
            print(f"\n[MEMORY STORAGE] New memory added to database via direct connection:\n  Content: {content}\n  User ID: {user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            self.connection.rollback()
            return None

    def search_memories(self, query, limit=5, user_id=None):
        """Search for memories similar to the query."""
        if not self.cursor:
            logger.error("Not connected to the database")
            return []

        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)

            # Search using vector similarity
            # Convert the embedding to a string representation for PostgreSQL
            embedding_str = '{' + ','.join(str(x) for x in query_embedding) + '}'

            # Prepare query and parameters
            if user_id:
                try:
                    # Try to use the user-specific function first
                    self.cursor.execute(
                        """
                        SELECT * FROM match_memories_by_user(%s::float[], %s, %s, %s);
                        """,
                        (embedding_str, 0.3, limit, user_id)
                    )
                    logger.info(f"Using match_memories_by_user with user_id: {user_id}")
                except Exception as e:
                    logger.error(f"Error using match_memories_by_user: {e}")
                    # Fall back to the generic function
                    self.cursor.execute(
                        """
                        SELECT * FROM match_memories(%s::float[], %s, %s);
                        """,
                        (embedding_str, 0.3, limit)
                    )
                    logger.info("Falling back to match_memories without user_id")
            else:
                self.cursor.execute(
                    """
                    SELECT * FROM match_memories(%s::float[], %s, %s);
                    """,
                    (embedding_str, 0.3, limit)
                )

            memories = self.cursor.fetchall()

            if not memories:
                logger.info("No similar memories found")
                return []

            logger.info(f"Found {len(memories)} similar memories")
            return memories
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id):
        """Delete a memory from the database."""
        if not self.cursor:
            logger.error("Not connected to the database")
            return False

        try:
            self.cursor.execute(
                "DELETE FROM memories WHERE id = %s RETURNING id;",
                (memory_id,)
            )
            result = self.cursor.fetchone()
            self.connection.commit()

            if result:
                logger.info(f"Deleted memory with ID: {memory_id}")
                return True
            else:
                logger.warning(f"Memory with ID {memory_id} not found")
                return False
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            self.connection.rollback()
            return False

    def clear_all_memories(self):
        """Clear all memories from the database."""
        if not self.cursor:
            logger.error("Not connected to the database")
            return False

        try:
            self.cursor.execute("SELECT clear_all_memories();")
            self.connection.commit()
            logger.info("All memories cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            self.connection.rollback()
            return False

    def close(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

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
        logger.error("Failed to connect to the database")
        return

    try:
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
    finally:
        # Close the database connection
        manager.close()

if __name__ == "__main__":
    main()
