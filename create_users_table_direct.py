import psycopg2
import logging
from dotenv import load_dotenv
import os
import requests
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def refresh_supabase_schema():
    """Refresh the Supabase schema cache by making a simple query to the table."""
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
        
        # Make a simple query to force schema refresh
        headers = {
            'apikey': supabase_key,
            'Authorization': f'Bearer {supabase_key}'
        }
        response = requests.get(f"{supabase_url}/rest/v1/users?select=id&limit=1", headers=headers)
        
        if response.status_code == 200:
            logger.info("Successfully refreshed Supabase schema cache")
            return True
        else:
            logger.error(f"Failed to refresh schema: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error refreshing schema: {e}")
        return False

def create_users_table():
    """Create the user_profiles table using direct PostgreSQL connection."""
    try:
        # Load environment variables from .env
        load_dotenv()

        # Fetch variables
        pg_user = os.getenv("user")
        pg_password = os.getenv("password")
        pg_host = os.getenv("host")
        pg_port = os.getenv("port")
        pg_database = os.getenv("dbname")

        # Check if PostgreSQL connection details are available
        if not pg_user or not pg_password or not pg_host or not pg_port or not pg_database:
            logger.error("PostgreSQL connection details not found in .env file")
            logger.info("Please make sure the .env file contains the following variables:")
            logger.info("  user: PostgreSQL username")
            logger.info("  password: PostgreSQL password")
            logger.info("  host: PostgreSQL host")
            logger.info("  port: PostgreSQL port")
            logger.info("  dbname: PostgreSQL database name")
            return False

        logger.info("PostgreSQL credentials loaded successfully")

        # Connect to PostgreSQL
        connection = psycopg2.connect(
            user=pg_user,
            password=pg_password,
            host=pg_host,
            port=pg_port,
            dbname=pg_database
        )
        logger.info("Connected to PostgreSQL successfully")

        # Create a cursor to execute SQL queries
        cursor = connection.cursor()

        # First, check if the table exists and print its structure
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'user_profiles'
        """)
        columns = cursor.fetchall()
        if columns:
            logger.info("Current user_profiles table structure:")
            for col in columns:
                logger.info(f"Column: {col[0]}, Type: {col[1]}")
        else:
            logger.info("user_profiles table does not exist yet")

        # Create user_profiles table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_profiles (
            id UUID PRIMARY KEY REFERENCES auth.users(id),
            username TEXT UNIQUE NOT NULL,
            display_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login TIMESTAMPTZ,
            preferences JSONB DEFAULT '{}'::JSONB
        );
        """

        cursor.execute(create_table_query)
        connection.commit()
        logger.info("user_profiles table created successfully")

        # Add user_id column to memories table if it doesn't exist
        alter_table_query = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'user_id'
            ) THEN
                ALTER TABLE memories ADD COLUMN user_id UUID REFERENCES auth.users(id);

                -- Create an index on user_id for faster queries
                CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories(user_id);
            END IF;
        END $$;
        """

        cursor.execute(alter_table_query)
        connection.commit()
        logger.info("Added user_id column to memories table (if it didn't exist)")

        # Close the cursor and connection
        cursor.close()
        connection.close()
        logger.info("Connection closed")

        # Refresh Supabase schema cache
        if refresh_supabase_schema():
            logger.info("Schema cache refreshed successfully")
        else:
            logger.warning("Failed to refresh schema cache, but table was created successfully")

        return True
    except Exception as e:
        logger.error(f"Error creating user_profiles table: {e}")
        return False

if __name__ == "__main__":
    success = create_users_table()
    if success:
        logger.info("Users table setup completed successfully")
    else:
        logger.error("Failed to set up users table")
