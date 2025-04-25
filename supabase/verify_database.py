import os
import json
import logging
from pathlib import Path
from supabase import create_client
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseVerifier:
    def __init__(self, verbose=False):
        self.supabase_url = None
        self.supabase_key = None
        self.supabase_client = None
        self.verbose = verbose
        self.issues = []

    def load_credentials(self):
        """Load Supabase credentials from .api_key.json file."""
        try:
            api_key_path = Path('.api_key.json')
            with open(api_key_path, 'r') as f:
                api_keys = json.load(f)
                self.supabase_url = api_keys.get('supabase_url')
                self.supabase_key = api_keys.get('supabase_api_key')

            if not self.supabase_url or not self.supabase_key:
                self.issues.append("Supabase credentials not found or incomplete in .api_key.json")
                logger.error("Supabase credentials not found or incomplete in .api_key.json")
                return False

            logger.info("Supabase credentials loaded successfully")
            return True
        except Exception as e:
            self.issues.append(f"Error loading Supabase credentials: {e}")
            logger.error(f"Error loading Supabase credentials: {e}")
            return False

    def connect_to_supabase(self):
        """Connect to Supabase using the loaded credentials."""
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase successfully")
            return True
        except Exception as e:
            self.issues.append(f"Failed to connect to Supabase: {e}")
            logger.error(f"Failed to connect to Supabase: {e}")
            return False

    def check_tables(self):
        """Check if required tables exist."""
        tables_to_check = ['memories', 'users']
        missing_tables = []
        
        for table in tables_to_check:
            try:
                response = self.supabase_client.table(table).select('*').limit(1).execute()
                logger.info(f"Table '{table}' exists")
            except Exception as e:
                missing_tables.append(table)
                self.issues.append(f"Table '{table}' does not exist or is not accessible")
                logger.error(f"Table '{table}' does not exist or is not accessible: {e}")
        
        if missing_tables:
            logger.error(f"Missing tables: {', '.join(missing_tables)}")
            return False
        
        return True

    def check_memories_columns(self):
        """Check if the memories table has the required columns."""
        try:
            # Query to get column information
            query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'memories'
            """
            response = self.supabase_client.rpc('query', {'query': query}).execute()
            
            if not response.data:
                self.issues.append("Could not retrieve column information for memories table")
                logger.error("Could not retrieve column information for memories table")
                return False
            
            # Check for required columns
            columns = {col['column_name']: col['data_type'] for col in response.data}
            
            required_columns = {
                'id': 'text',
                'content': 'text',
                'embedding': 'USER-DEFINED',  # VECTOR type
                'importance': 'integer',
                'category': 'text',
                'timestamp': 'timestamp with time zone',
                'user_id': 'text'
            }
            
            missing_columns = []
            for col, dtype in required_columns.items():
                if col not in columns:
                    missing_columns.append(col)
                    self.issues.append(f"Required column '{col}' is missing from memories table")
                
            if missing_columns:
                logger.error(f"Missing columns in memories table: {', '.join(missing_columns)}")
                return False
            
            # Check for created_at column (potential source of issues)
            if 'created_at' in columns:
                logger.warning("The 'created_at' column exists in memories table, which may cause conflicts with 'timestamp'")
                self.issues.append("The 'created_at' column exists in memories table, which may cause conflicts with 'timestamp'")
            
            logger.info("All required columns exist in memories table")
            return True
        except Exception as e:
            self.issues.append(f"Error checking memories table columns: {e}")
            logger.error(f"Error checking memories table columns: {e}")
            return False

    def check_functions(self):
        """Check if required functions exist and have the correct signature."""
        functions_to_check = [
            'match_memories',
            'match_memories_by_user',
            'clear_all_memories'
        ]
        
        missing_functions = []
        
        for func in functions_to_check:
            try:
                # Query to check if function exists
                query = f"""
                SELECT pg_get_functiondef(oid) 
                FROM pg_proc 
                WHERE proname = '{func}'
                """
                response = self.supabase_client.rpc('query', {'query': query}).execute()
                
                if not response.data:
                    missing_functions.append(func)
                    self.issues.append(f"Function '{func}' does not exist")
                    logger.error(f"Function '{func}' does not exist")
                else:
                    function_def = response.data[0]['pg_get_functiondef']
                    
                    # Check for potential issues in function definitions
                    if func in ['match_memories', 'match_memories_by_user']:
                        if 'created_at' in function_def and 'timestamp' not in function_def:
                            self.issues.append(f"Function '{func}' uses 'created_at' column but not 'timestamp'")
                            logger.warning(f"Function '{func}' uses 'created_at' column but not 'timestamp'")
                        
                        if 'm.created_at' in function_def:
                            self.issues.append(f"Function '{func}' directly references 'm.created_at' which may cause errors")
                            logger.warning(f"Function '{func}' directly references 'm.created_at' which may cause errors")
                    
                    if self.verbose:
                        logger.info(f"Function '{func}' definition: {function_def[:100]}...")
            
            except Exception as e:
                self.issues.append(f"Error checking function '{func}': {e}")
                logger.error(f"Error checking function '{func}': {e}")
        
        if missing_functions:
            logger.error(f"Missing functions: {', '.join(missing_functions)}")
            return False
        
        logger.info("All required functions exist")
        return True

    def test_function_execution(self):
        """Test if the functions can be executed successfully."""
        try:
            # Test match_memories function
            response = self.supabase_client.rpc('match_memories', {
                'query_embedding': [0.1] * 300,  # Dummy embedding
                'match_threshold': 0.5,
                'match_count': 1
            }).execute()
            logger.info("Successfully executed match_memories function")
            
            # Test match_memories_by_user function
            response = self.supabase_client.rpc('match_memories_by_user', {
                'query_embedding': [0.1] * 300,  # Dummy embedding
                'match_threshold': 0.5,
                'match_count': 1,
                'user_id_param': 'test_user'  # Test user ID
            }).execute()
            logger.info("Successfully executed match_memories_by_user function")
            
            return True
        except Exception as e:
            self.issues.append(f"Error executing memory functions: {e}")
            logger.error(f"Error executing memory functions: {e}")
            return False

    def verify(self):
        """Run all verification checks."""
        if not self.load_credentials():
            return False
        
        if not self.connect_to_supabase():
            return False
        
        # Run all checks
        tables_ok = self.check_tables()
        columns_ok = self.check_memories_columns()
        functions_ok = self.check_functions()
        execution_ok = self.test_function_execution()
        
        # Overall status
        if tables_ok and columns_ok and functions_ok and execution_ok:
            logger.info("Database verification completed successfully - no issues found")
            return True
        else:
            logger.warning("Database verification completed with issues")
            return False
    
    def print_report(self):
        """Print a report of the verification results."""
        print("\n=== Database Verification Report ===")
        
        if not self.issues:
            print("✅ All checks passed! Your database setup is correct.")
        else:
            print(f"❌ Found {len(self.issues)} issues with your database setup:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
            
            print("\nRecommended actions:")
            print("1. Run the database initialization script: python supabase/initialize_database.py")
            print("2. If issues persist, manually execute the SQL in supabase/standard_schema.sql")
            print("3. For detailed instructions, see supabase/SETUP_INSTRUCTIONS.md")

def main():
    parser = argparse.ArgumentParser(description="Verify Supabase database setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    verifier = DatabaseVerifier(verbose=args.verbose)
    verifier.verify()
    verifier.print_report()

if __name__ == "__main__":
    main()
