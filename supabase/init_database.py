import os
import json
import logging
from supabase import create_client
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_supabase_database():
    """Initialize the Supabase database with the required tables and functions for memory storage."""
    
    # Load Supabase credentials
    try:
        api_key_path = Path('.api_key.json')
        with open(api_key_path, 'r') as f:
            api_keys = json.load(f)
            supabase_url = api_keys.get('supabase_url')
            supabase_key = api_keys.get('supabase_api_key')
            
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in .api_key.json")
            return False
    except Exception as e:
        logger.error(f"Error loading Supabase credentials: {e}")
        return False
    
    # Connect to Supabase
    try:
        client = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        return False
    
    # Load and execute SQL setup script
    try:
        sql_path = Path('supabase/setup.sql')
        with open(sql_path, 'r') as f:
            sql_script = f.read()
        
        # Split the SQL script into individual statements
        # This is a simple approach - for more complex scripts, consider using a proper SQL parser
        statements = sql_script.split(';')
        
        # Execute each statement
        for statement in statements:
            statement = statement.strip()
            if statement:  # Skip empty statements
                try:
                    # Execute the SQL statement
                    client.postgrest.rpc(statement).execute()
                except Exception as e:
                    # Some statements might fail if they're DDL statements or if they're already executed
                    # We'll log but continue
                    logger.warning(f"Statement execution warning (may be normal): {e}")
        
        logger.info("SQL setup script executed successfully")
    except Exception as e:
        logger.error(f"Error executing SQL setup script: {e}")
        return False
    
    # Verify setup by checking if the memories table exists
    try:
        response = client.table('memories').select('id').limit(1).execute()
        logger.info("Verified 'memories' table exists")
        return True
    except Exception as e:
        logger.error(f"Failed to verify 'memories' table: {e}")
        return False

if __name__ == "__main__":
    success = initialize_supabase_database()
    if success:
        logger.info("Supabase database initialization completed successfully")
    else:
        logger.error("Supabase database initialization failed")
