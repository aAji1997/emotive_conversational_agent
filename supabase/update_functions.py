import os
import json
import logging
from supabase import create_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_credentials():
    """Load Supabase credentials from .api_key.json file."""
    try:
        with open('.api_key.json', 'r') as f:
            api_keys = json.load(f)
            supabase_url = api_keys.get('supabase_url')
            supabase_key = api_keys.get('supabase_api_key')

        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found or incomplete in .api_key.json")
            return None, None

        logger.info("Supabase credentials loaded successfully")
        return supabase_url, supabase_key
    except Exception as e:
        logger.error(f"Error loading Supabase credentials: {e}")
        return None, None

def connect_to_supabase(url, key):
    """Connect to Supabase using the loaded credentials."""
    try:
        client = create_client(url, key)
        logger.info("Connected to Supabase successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return None

def execute_sql_file(client, file_path):
    """Execute the SQL file to update the functions."""
    try:
        # Load SQL file
        with open(file_path, 'r') as f:
            sql_script = f.read()

        # Split the SQL script into individual statements
        statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]

        # Execute each statement
        for i, statement in enumerate(statements):
            try:
                # Execute SQL statement
                client.postgrest.rpc('exec_sql', {'query': statement}).execute()
                logger.info(f"Executed SQL statement {i+1}/{len(statements)}")
            except Exception as e:
                logger.warning(f"Warning executing SQL statement {i+1}: {e}")

        logger.info("SQL update completed")
        return True
    except Exception as e:
        logger.error(f"Error executing SQL update: {e}")
        return False

def verify_functions(client):
    """Verify that the functions were updated successfully."""
    try:
        # Check if the match_memories function exists
        try:
            # This will fail if the function doesn't exist
            client.rpc('match_memories', {
                'query_embedding': [0.1] * 300,  # Dummy embedding
                'match_threshold': 0.5,
                'match_count': 1
            }).execute()
            logger.info("Verified 'match_memories' function exists")

            # Check if the match_memories_by_user function exists
            try:
                # This will fail if the function doesn't exist
                client.rpc('match_memories_by_user', {
                    'query_embedding': [0.1] * 300,  # Dummy embedding
                    'match_threshold': 0.5,
                    'match_count': 1,
                    'user_id_param': '00000000-0000-0000-0000-000000000000'  # Dummy user ID
                }).execute()
                logger.info("Verified 'match_memories_by_user' function exists")
                return True
            except Exception as e:
                logger.error(f"Could not verify 'match_memories_by_user' function: {e}")
                return False
        except Exception as e:
            logger.error(f"Could not verify 'match_memories' function: {e}")
            return False
    except Exception as e:
        logger.error(f"Failed to verify functions: {e}")
        return False

def main():
    """Main function to update the SQL functions."""
    # Load credentials
    supabase_url, supabase_key = load_credentials()
    if not supabase_url or not supabase_key:
        return False

    # Connect to Supabase
    client = connect_to_supabase(supabase_url, supabase_key)
    if not client:
        return False

    # Execute SQL file
    if not execute_sql_file(client, 'supabase/update_memory_functions.sql'):
        logger.warning("SQL update had some issues, but continuing...")

    # Verify functions
    if not verify_functions(client):
        logger.warning("Function verification had issues")
        return False

    logger.info("SQL functions updated successfully")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ SQL functions updated successfully!")
    else:
        print("\n❌ SQL function update failed.")
        print("Please check the logs for details and try again.")
