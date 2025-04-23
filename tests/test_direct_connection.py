import os
import logging
import psycopg2
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test the direct connection to the PostgreSQL database."""
    try:
        # Load environment variables
        load_dotenv()
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        dbname = os.getenv("dbname")
        
        if not all([user, password, host, port, dbname]):
            logger.error("Missing required environment variables in .env file")
            return False
        
        # Connect to the database
        connection = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            dbname=dbname
        )
        cursor = connection.cursor()
        
        # Test the connection with a simple query
        cursor.execute("SELECT NOW();")
        result = cursor.fetchone()
        logger.info(f"Database connection successful. Current time: {result[0]}")
        
        # Check if the pgvector extension is enabled
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_extension = cursor.fetchone()
        
        if vector_extension:
            logger.info("pgvector extension is enabled")
        else:
            logger.warning("pgvector extension is not enabled")
            
            # Try to enable the extension
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                connection.commit()
                logger.info("Successfully enabled pgvector extension")
            except Exception as e:
                logger.error(f"Failed to enable pgvector extension: {e}")
                connection.rollback()
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
        logger.info("Database connection closed")
        
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    
    if success:
        print("\n✅ Database connection test successful!")
        print("You can now proceed with setting up the memory storage system.")
    else:
        print("\n❌ Database connection test failed.")
        print("Please check your .env file and make sure the database credentials are correct.")
