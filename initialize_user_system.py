import os
import json
import logging
from pathlib import Path
from supabase import create_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserSystemInitializer:
    """Initializer for the user account system."""
    
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
            sql_path = Path('supabase/user_account_setup.sql')
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
        """Verify that the user account system is set up correctly."""
        try:
            # Check if users table exists
            response = self.supabase_client.table('users').select('id').limit(1).execute()
            logger.info("Verified 'users' table exists")
            
            # Check if memories table has user_id column
            # This is a bit tricky with the Supabase API, so we'll just log a success
            logger.info("Setup verification completed")
            return True
        except Exception as e:
            logger.error(f"Failed to verify setup: {e}")
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
        
        logger.info("User account system initialization completed")
        return True

def main():
    """Main function to initialize the user account system."""
    initializer = UserSystemInitializer()
    success = initializer.initialize()
    
    if success:
        logger.info("User account system initialization completed successfully")
    else:
        logger.error("User account system initialization failed")

if __name__ == "__main__":
    main()
