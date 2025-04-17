import json
import os
import psycopg2
from dotenv import load_dotenv

def setup_database():
    """Set up the Supabase database with the necessary tables and functions."""
    # Load environment variables from .env
    load_dotenv()

    # Try to load credentials from .api_key.json first
    try:
        with open(".api_key.json", "r") as f:
            api_keys = json.load(f)
            supabase_url = api_keys.get("supabase_url")
            supabase_key = api_keys.get("supabase_api_key")
            
            # Extract connection details from the URL
            # Format: postgres://[user]:[password]@[host]:[port]/[dbname]
            if supabase_url and "://" in supabase_url:
                # Remove postgres:// prefix
                conn_string = supabase_url.split("://")[1]
                # Split user:password@host:port/dbname
                user_pass, host_port_db = conn_string.split("@")
                user, password = user_pass.split(":")
                host_port, dbname = host_port_db.split("/")
                host, port = host_port.split(":")
                
                # Set environment variables
                os.environ["user"] = user
                os.environ["password"] = password
                os.environ["host"] = host
                os.environ["port"] = port
                os.environ["dbname"] = dbname
                
                print("Credentials loaded from .api_key.json")
    except Exception as e:
        print(f"Could not load credentials from .api_key.json: {e}")
        print("Falling back to .env file")

    # Fetch variables from environment
    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    if not all([USER, PASSWORD, HOST, PORT, DBNAME]):
        print("Missing database credentials. Please check your .env file or .api_key.json")
        return False

    # Connect to the database
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        print("Connection successful!")
        
        # Create a cursor to execute SQL queries
        cursor = connection.cursor()
        
        # Read the SQL file
        with open("supabase/setup_database.sql", "r") as f:
            sql = f.read()
        
        # Execute the SQL
        cursor.execute(sql)
        connection.commit()
        print("Database setup complete!")

        # Close the cursor and connection
        cursor.close()
        connection.close()
        print("Connection closed.")
        return True

    except Exception as e:
        print(f"Failed to set up database: {e}")
        return False

if __name__ == "__main__":
    setup_database()
