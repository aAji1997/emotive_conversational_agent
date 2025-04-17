import psycopg2
import logging
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_users_table():
    """Create the users table using direct PostgreSQL connection."""
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

        # Create users table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            display_name TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login TIMESTAMPTZ,
            preferences JSONB DEFAULT '{}'::JSONB
        );
        """

        cursor.execute(create_table_query)
        connection.commit()
        logger.info("Users table created successfully")

        # Add user_id column to memories table if it doesn't exist
        alter_table_query = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'memories' AND column_name = 'user_id'
            ) THEN
                ALTER TABLE memories ADD COLUMN user_id TEXT REFERENCES users(id);

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

        return True
    except Exception as e:
        logger.error(f"Error creating users table: {e}")
        return False

if __name__ == "__main__":
    success = create_users_table()
    if success:
        logger.info("Users table setup completed successfully")
    else:
        logger.error("Failed to set up users table")
