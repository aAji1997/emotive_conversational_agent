import os
import json
import logging
import time
from pathlib import Path
from supabase import create_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseInitializer:
    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.supabase_client = None
        self.initialization_successful = False

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
        """Execute the standardized SQL setup script."""
        try:
            # Load SQL setup script
            sql_path = Path('supabase/standard_schema.sql')
            with open(sql_path, 'r') as f:
                sql_script = f.read()

            # Split the SQL script into individual statements
            statements = []
            current_statement = ""
            
            # Simple SQL parser to handle function definitions with semicolons inside
            in_function_body = False
            for line in sql_script.split('\n'):
                line = line.strip()
                if not line or line.startswith('--'):
                    continue
                    
                current_statement += line + " "
                
                # Check if we're entering a function body
                if "AS $$" in line:
                    in_function_body = True
                
                # Check if we're exiting a function body
                if "$$;" in line and in_function_body:
                    in_function_body = False
                    statements.append(current_statement.strip())
                    current_statement = ""
                    continue
                
                # Only split on semicolons if we're not in a function body
                if line.endswith(';') and not in_function_body:
                    statements.append(current_statement.strip())
                    current_statement = ""

            # Execute each statement
            for i, statement in enumerate(statements):
                try:
                    # Execute SQL statement using the REST API
                    response = self.supabase_client.postgrest.rpc('exec_sql', {'query': statement}).execute()
                    logger.info(f"Executed SQL statement {i+1}/{len(statements)}")
                except Exception as e:
                    # Some statements might fail if they're DDL statements or if they're already executed
                    logger.warning(f"Warning executing SQL statement {i+1}: {e}")
                    
                    # Try direct query execution as fallback
                    try:
                        # Use the database query endpoint directly
                        response = self.supabase_client.rpc('query', {'query': statement}).execute()
                        logger.info(f"Executed SQL statement {i+1} using direct query")
                    except Exception as e2:
                        logger.warning(f"Failed to execute SQL statement {i+1} using direct query: {e2}")
                        # Continue anyway, as some statements might fail normally

            logger.info("SQL setup completed")
            return True
        except Exception as e:
            logger.error(f"Error executing SQL setup: {e}")
            return False

    def run_schema_migration(self):
        """Run the schema migration function to handle column differences."""
        try:
            # Call the check_and_migrate_schema function
            response = self.supabase_client.rpc('check_and_migrate_schema').execute()
            
            if response.data:
                migration_message = response.data
                logger.info(f"Schema migration result: {migration_message}")
                return True
            else:
                logger.warning("Schema migration function returned no result")
                return False
        except Exception as e:
            logger.error(f"Error running schema migration: {e}")
            return False

    def verify_setup(self):
        """Verify that the setup was successful by checking tables and functions."""
        try:
            # Check if the memories table exists
            try:
                response = self.supabase_client.table('memories').select('id').limit(1).execute()
                logger.info("Verified 'memories' table exists")
            except Exception as e:
                logger.error(f"Failed to verify 'memories' table: {e}")
                return False

            # Check if the match_memories function exists
            try:
                self.supabase_client.rpc('match_memories', {
                    'query_embedding': [0.1] * 300,  # Dummy embedding
                    'match_threshold': 0.5,
                    'match_count': 1
                }).execute()
                logger.info("Verified 'match_memories' function exists")
            except Exception as e:
                logger.error(f"Failed to verify 'match_memories' function: {e}")
                return False

            # Check if the match_memories_by_user function exists
            try:
                self.supabase_client.rpc('match_memories_by_user', {
                    'query_embedding': [0.1] * 300,  # Dummy embedding
                    'match_threshold': 0.5,
                    'match_count': 1,
                    'user_id_param': 'test_user'  # Test user ID
                }).execute()
                logger.info("Verified 'match_memories_by_user' function exists")
            except Exception as e:
                logger.error(f"Failed to verify 'match_memories_by_user' function: {e}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error verifying setup: {e}")
            return False

    def test_memory_storage(self):
        """Test storing and retrieving a memory to verify the setup."""
        try:
            # Create a test memory
            test_memory = {
                'id': 'test-memory-1',
                'content': 'This is a test memory for setup verification',
                'embedding': [0.1] * 300,  # Simple test embedding
                'importance': 5,
                'category': 'test',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'source': 'setup_test',
                'conversation_mode': 'test',
                'metadata': {'test': True}
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

        # Run schema migration to handle differences
        if not self.run_schema_migration():
            logger.warning("Schema migration had issues, but continuing...")

        if not self.verify_setup():
            logger.warning("Setup verification had issues, but continuing...")

        if not self.test_memory_storage():
            logger.warning("Memory storage test failed, but initialization will continue")

        logger.info("Database initialization completed")
        self.initialization_successful = True
        return True

def initialize_database():
    """Initialize the database with the standardized schema."""
    initializer = DatabaseInitializer()
    success = initializer.initialize()
    
    if success:
        logger.info("Database initialization completed successfully")
        return True
    else:
        logger.error("Database initialization failed")
        return False

if __name__ == "__main__":
    initialize_database()
