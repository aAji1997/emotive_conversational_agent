import os
import json
import logging
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def execute_sql_setup():
    """Execute the SQL setup script directly using the Supabase REST API."""
    
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
    
    # Load SQL setup script
    try:
        sql_path = Path('supabase/setup.sql')
        with open(sql_path, 'r') as f:
            sql_script = f.read()
    except Exception as e:
        logger.error(f"Error loading SQL script: {e}")
        return False
    
    # Execute SQL script via REST API
    try:
        # Headers for authentication
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Endpoint for SQL execution
        sql_url = f"{supabase_url}/rest/v1/rpc/exec_sql"
        
        # Split the SQL script into individual statements
        statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
        
        # Execute each statement
        for i, statement in enumerate(statements):
            try:
                # Prepare the payload
                payload = {
                    "query": statement
                }
                
                # Make the request
                response = requests.post(sql_url, headers=headers, json=payload)
                
                # Check response
                if response.status_code >= 400:
                    logger.warning(f"Statement {i+1} execution warning: {response.text}")
                else:
                    logger.info(f"Statement {i+1} executed successfully")
            except Exception as e:
                logger.warning(f"Error executing statement {i+1}: {e}")
        
        logger.info("SQL setup script execution completed")
        return True
    except Exception as e:
        logger.error(f"Error during SQL execution: {e}")
        return False

if __name__ == "__main__":
    success = execute_sql_setup()
    if success:
        logger.info("SQL setup execution completed")
    else:
        logger.error("SQL setup execution failed")
