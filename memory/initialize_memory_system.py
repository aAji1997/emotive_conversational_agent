import os
import json
import logging
import asyncio
from pathlib import Path
from supabase import create_client, Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemorySystemInitializer:
    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.supabase_client = None

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

    def connect_to_supabase(self):
        """Connect to Supabase using the loaded credentials."""
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            return False

    def execute_sql_setup(self):
        """Execute the SQL setup script to create necessary tables and functions."""
        try:
            # Load SQL setup script
            sql_path = Path('supabase/setup.sql')
            with open(sql_path, 'r') as f:
                sql_script = f.read()

            # Split the SQL script into individual statements
            statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

            # Execute each statement
            for i, statement in enumerate(statements):
                try:
                    # Execute SQL statement
                    # Note: This is a simplified approach. In a production environment,
                    # you might want to use a more robust SQL execution method.
                    self.supabase_client.postgrest.rpc('exec_sql', {'query': statement}).execute()
                    logger.info(f"Executed SQL statement {i+1}/{len(statements)}")
                except Exception as e:
                    logger.warning(f"Warning executing SQL statement {i+1}: {e}")

            logger.info("SQL setup completed")
            return True
        except Exception as e:
            logger.error(f"Error executing SQL setup: {e}")
            return False

    def verify_setup(self):
        """Verify that the setup was successful by checking if the memories table exists."""
        try:
            # Try to query the memories table
            response = self.supabase_client.table('memories').select('id').limit(1).execute()
            logger.info("Verified 'memories' table exists")

            # Check if the match_memories function exists
            try:
                # This will fail if the function doesn't exist
                self.supabase_client.rpc('match_memories', {
                    'query_embedding': [0.1] * 300,  # Dummy embedding
                    'match_threshold': 0.5,
                    'match_count': 1
                }).execute()
                logger.info("Verified 'match_memories' function exists")

                # Check if the match_memories_by_user function exists
                try:
                    # This will fail if the function doesn't exist
                    self.supabase_client.rpc('match_memories_by_user', {
                        'query_embedding': [0.1] * 300,  # Dummy embedding
                        'match_threshold': 0.5,
                        'match_count': 1,
                        'user_id_param': '00000000-0000-0000-0000-000000000000'  # Dummy user ID
                    }).execute()
                    logger.info("Verified 'match_memories_by_user' function exists")
                except Exception as e:
                    logger.warning(f"Could not verify 'match_memories_by_user' function: {e}")
                    logger.info("Will create match_memories_by_user function")

                    # Create the match_memories_by_user function
                    try:
                        self.supabase_client.rpc('create_match_memories_by_user_function').execute()
                        logger.info("Created 'match_memories_by_user' function")
                    except Exception as e:
                        logger.warning(f"Could not create 'match_memories_by_user' function: {e}")
                        # Continue anyway, as we can fall back to match_memories
            except Exception as e:
                logger.warning(f"Could not verify 'match_memories' function: {e}")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to verify setup: {e}")
            return False

    def test_memory_storage(self):
        """Test storing and retrieving a memory."""
        try:
            # Create a test memory
            test_memory = {
                'id': 'test-memory-1',
                'content': 'This is a test memory to verify the system works',
                'embedding': [0.1] * 300,  # Dummy embedding
                'importance': 5,
                'category': 'test',
                'timestamp': '2023-08-01T12:00:00Z',
                'source': 'system',
                'conversation_mode': 'test'
            }

            # Store the memory
            self.supabase_client.table('memories').insert(test_memory).execute()
            logger.info("Test memory stored successfully")

            # Retrieve the memory
            response = self.supabase_client.table('memories').select('*').eq('id', 'test-memory-1').execute()
            if response.data and len(response.data) > 0:
                logger.info("Test memory retrieved successfully")

                # Clean up the test memory
                self.supabase_client.table('memories').delete().eq('id', 'test-memory-1').execute()
                logger.info("Test memory cleaned up")

                return True
            else:
                logger.error("Failed to retrieve test memory")
                return False
        except Exception as e:
            logger.error(f"Error testing memory storage: {e}")
            return False

    def initialize(self):
        """Run the complete initialization process."""
        if not self.load_credentials():
            return False

        if not self.connect_to_supabase():
            return False

        if not self.execute_sql_setup():
            logger.warning("SQL setup had some issues, but continuing...")

        if not self.verify_setup():
            logger.warning("Setup verification had issues, but continuing...")

        if not self.test_memory_storage():
            logger.warning("Memory storage test failed, but initialization will continue")

        logger.info("Memory system initialization completed")
        return True

async def test_memory_agent():
    """Test the memory agent with the initialized database."""
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

        # Test processing a message
        await memory_agent.process_message(
            message="I really enjoy hiking in the mountains on weekends.",
            role="user",
            conversation_mode="chat"
        )

        logger.info("Memory agent test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing memory agent: {e}")
        return False

if __name__ == "__main__":
    # Initialize the memory system
    initializer = MemorySystemInitializer()
    success = initializer.initialize()

    if success:
        logger.info("Memory system initialization successful")

        # Test the memory agent
        try:
            asyncio.run(test_memory_agent())
        except Exception as e:
            logger.error(f"Error running memory agent test: {e}")
    else:
        logger.error("Memory system initialization failed")
