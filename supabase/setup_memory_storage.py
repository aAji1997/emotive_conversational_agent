import psycopg2
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryStorageSetup:
    def __init__(self):
        """Initialize the memory storage setup."""
        self.connection = None
        self.cursor = None
        self.load_env_variables()

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

    def connect_to_database(self):
        """Connect to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                dbname=self.dbname
            )
            self.cursor = self.connection.cursor()
            logger.info("Connected to the database successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to the database: {e}")
            return False

    def execute_sql_command(self, sql_command, commit=True):
        """Execute a SQL command."""
        try:
            self.cursor.execute(sql_command)
            if commit:
                self.connection.commit()
            logger.info("SQL command executed successfully")
            return True
        except Exception as e:
            logger.error(f"Error executing SQL command: {e}")
            logger.error(f"Command: {sql_command[:100]}...")  # Log first 100 chars of the command
            self.connection.rollback()
            return False

    def execute_sql_file(self, file_path):
        """Execute SQL commands from a file."""
        try:
            with open(file_path, 'r') as f:
                sql_script = f.read()

            # Split the script into individual commands
            # This is a simple approach - for more complex scripts, consider using a proper SQL parser
            commands = sql_script.split(';')

            success_count = 0
            total_commands = len([cmd for cmd in commands if cmd.strip()])

            for i, command in enumerate(commands):
                command = command.strip()
                if command:  # Skip empty commands
                    try:
                        self.cursor.execute(command)
                        self.connection.commit()
                        success_count += 1
                        logger.info(f"Command {i+1}/{total_commands} executed successfully")
                    except Exception as e:
                        logger.warning(f"Warning executing command {i+1}/{total_commands}: {e}")
                        self.connection.rollback()

            logger.info(f"SQL file execution completed: {success_count}/{total_commands} commands successful")
            return success_count > 0
        except Exception as e:
            logger.error(f"Error executing SQL file: {e}")
            return False

    def enable_vector_extension(self):
        """Enable the pgvector extension."""
        return self.execute_sql_command("CREATE EXTENSION IF NOT EXISTS vector;")

    def create_memories_table(self):
        """Create the memories table."""
        sql_command = """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(300),
            importance INTEGER CHECK (importance >= 1 AND importance <= 10),
            category TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            source TEXT,
            conversation_mode TEXT,
            metadata JSONB DEFAULT '{}'::JSONB
        );
        """
        return self.execute_sql_command(sql_command)

    def create_vector_index(self):
        """Create an index on the embedding column for faster similarity searches."""
        sql_command = """
        CREATE INDEX IF NOT EXISTS memories_embedding_idx
        ON memories USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        return self.execute_sql_command(sql_command)

    def create_match_memories_function(self):
        """Create a function to match memories based on embedding similarity."""
        sql_command = """
        CREATE OR REPLACE FUNCTION match_memories(
            query_embedding FLOAT[],
            match_threshold FLOAT,
            match_count INT
        )
        RETURNS TABLE (
            id TEXT,
            content TEXT,
            importance INTEGER,
            category TEXT,
            "timestamp" TIMESTAMPTZ,
            source TEXT,
            conversation_mode TEXT,
            metadata JSONB,
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                m.id,
                m.content,
                m.importance,
                m.category,
                m.timestamp,
                m.source,
                m.conversation_mode,
                m.metadata,
                1 - (m.embedding <=> query_embedding::vector) AS similarity
            FROM
                memories m
            WHERE
                1 - (m.embedding <=> query_embedding::vector) > match_threshold
            ORDER BY
                m.embedding <=> query_embedding::vector
            LIMIT
                match_count;
        END;
        $$;
        """
        return self.execute_sql_command(sql_command)

    def create_clear_memories_function(self):
        """Create a function to clear all memories."""
        sql_command = """
        CREATE OR REPLACE FUNCTION clear_all_memories()
        RETURNS VOID
        LANGUAGE plpgsql
        AS $$
        BEGIN
            DELETE FROM memories;
        END;
        $$;
        """
        return self.execute_sql_command(sql_command)

    def verify_setup(self):
        """Verify that the setup was successful."""
        try:
            # Check if the memories table exists
            self.cursor.execute("SELECT to_regclass('public.memories');")
            result = self.cursor.fetchone()
            if result[0] is None:
                logger.error("Memories table does not exist")
                return False

            # Check if the match_memories function exists
            self.cursor.execute("SELECT proname FROM pg_proc WHERE proname = 'match_memories';")
            result = self.cursor.fetchone()
            if result is None:
                logger.error("match_memories function does not exist")
                return False

            logger.info("Setup verification successful")
            return True
        except Exception as e:
            logger.error(f"Error verifying setup: {e}")
            return False

    def insert_test_memory(self):
        """Insert a test memory to verify the system works."""
        try:
            from datetime import datetime

            # Create a test memory
            sql_command = """
            INSERT INTO memories (id, content, embedding, importance, category, timestamp, source, conversation_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
            """

            # Create a dummy embedding (300 dimensions)
            dummy_embedding = f"[{','.join(['0.1' for _ in range(300)])}]"

            values = (
                'test-memory-1',
                'This is a test memory to verify the system works',
                dummy_embedding,
                5,
                'test',
                datetime.now().isoformat(),
                'system',
                'test'
            )

            self.cursor.execute(sql_command, values)
            self.connection.commit()

            # Verify the memory was inserted
            self.cursor.execute("SELECT id FROM memories WHERE id = 'test-memory-1';")
            result = self.cursor.fetchone()

            if result:
                logger.info("Test memory inserted successfully")

                # Clean up the test memory
                self.cursor.execute("DELETE FROM memories WHERE id = 'test-memory-1';")
                self.connection.commit()
                logger.info("Test memory cleaned up")

                return True
            else:
                logger.error("Failed to insert test memory")
                return False
        except Exception as e:
            logger.error(f"Error inserting test memory: {e}")
            self.connection.rollback()
            return False

    def close_connection(self):
        """Close the database connection."""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Database connection closed")
            return True
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
            return False

    def setup(self):
        """Run the complete setup process."""
        if not self.connect_to_database():
            return False

        try:
            # Enable the pgvector extension
            if not self.enable_vector_extension():
                logger.error("Failed to enable pgvector extension")
                return False

            # Create the memories table
            if not self.create_memories_table():
                logger.error("Failed to create memories table")
                return False

            # Create the vector index
            if not self.create_vector_index():
                logger.warning("Failed to create vector index, but continuing...")

            # Create the match_memories function
            if not self.create_match_memories_function():
                logger.error("Failed to create match_memories function")
                return False

            # Create the clear_memories function
            if not self.create_clear_memories_function():
                logger.warning("Failed to create clear_memories function, but continuing...")

            # Verify the setup
            if not self.verify_setup():
                logger.error("Setup verification failed")
                return False

            # Insert a test memory
            if not self.insert_test_memory():
                logger.warning("Test memory insertion failed, but continuing...")

            logger.info("Memory storage setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            return False
        finally:
            self.close_connection()

if __name__ == "__main__":
    setup = MemoryStorageSetup()
    success = setup.setup()

    if success:
        logger.info("Memory storage setup completed successfully")
        print("\n✅ Memory storage setup completed successfully!")
        print("You can now use the memory system in your conversational agent.")
    else:
        logger.error("Memory storage setup failed")
        print("\n❌ Memory storage setup failed.")
        print("Please check the logs for details and try again.")
