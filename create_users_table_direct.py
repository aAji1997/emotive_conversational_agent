import os
import json
import logging
from pathlib import Path
from supabase import create_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_users_table():
    """Create the users table using Supabase client."""
    try:
        # Load Supabase credentials
        api_key_path = Path('.api_key.json')
        with open(api_key_path, 'r') as f:
            api_keys = json.load(f)
            supabase_url = api_keys.get('supabase_url')
            supabase_key = api_keys.get('supabase_api_key')
            
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in .api_key.json")
            return False
            
        # Connect to Supabase
        client = create_client(supabase_url, supabase_key)
        logger.info("Connected to Supabase successfully")
        
        # Check if users table exists
        try:
            client.table('users').select('id').limit(1).execute()
            logger.info("Users table already exists")
            return True
        except Exception as e:
            logger.info("Users table does not exist, creating it...")
        
        # Create users table using Supabase client
        # First, create a test user to trigger table creation
        test_user = {
            'id': 'test_user_id',
            'username': 'test_user',
            'display_name': 'Test User',
            'created_at': '2025-04-22T00:00:00Z',
            'last_login': '2025-04-22T00:00:00Z',
            'preferences': {}
        }
        
        # Insert test user
        client.table('users').insert(test_user).execute()
        logger.info("Users table created successfully")
        
        # Delete test user
        client.table('users').delete().eq('id', 'test_user_id').execute()
        logger.info("Test user removed")
        
        return True
    except Exception as e:
        logger.error(f"Error creating users table: {e}")
        return False

if __name__ == "__main__":
    create_users_table()
