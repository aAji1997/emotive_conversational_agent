import os
import json
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from supabase import create_client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserAccountManager:
    """Manager for user account operations.

    This class uses a singleton pattern to ensure only one instance is created.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UserAccountManager, cls).__new__(cls)
            cls._instance.supabase_url = None
            cls._instance.supabase_key = None
            cls._instance.supabase_client = None
            cls._instance.current_user = None
        return cls._instance

    def __init__(self):
        # Skip initialization if already initialized
        if UserAccountManager._initialized:
            return
        UserAccountManager._initialized = True

    def load_credentials(self) -> bool:
        """Load Supabase credentials from .api_key.json file."""
        # If credentials are already loaded, return True
        if self.supabase_url and self.supabase_key:
            logger.info("Supabase credentials already loaded, skipping load")
            return True

        try:
            # Add more detailed logging
            import traceback
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_info = inspect.getframeinfo(caller_frame)
            logger.info(f"Loading Supabase credentials from {caller_info.filename}:{caller_info.lineno}")

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
            logger.error(traceback.format_exc())
            return False

    def connect(self) -> bool:
        """Connect to Supabase."""
        # If already connected with a real client, return True
        if self.supabase_client is not None and not isinstance(self.supabase_client, str):
            logger.info("Already connected to Supabase with a valid client, skipping connection")
            return True

        # Always load credentials to ensure they're available
        logger.info("Loading Supabase credentials...")
        if not self.load_credentials():
            logger.error("Failed to load Supabase credentials")
            return False

        logger.info(f"Credentials loaded - URL: {self.supabase_url[:10]}... Key: {self.supabase_key[:10]}...")

        try:
            # Add more detailed logging
            import traceback
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_info = inspect.getframeinfo(caller_frame)
            logger.info(f"Connecting to Supabase from {caller_info.filename}:{caller_info.lineno}")

            # Create a new client regardless of environment variable
            logger.info("Creating Supabase client...")
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase successfully")

            # Test the connection with a simple query
            try:
                logger.info("Testing Supabase connection with a simple query...")
                response = self.supabase_client.table('users').select('count(*)', count='exact').execute()
                user_count = response.count if hasattr(response, 'count') else 'unknown'
                logger.info(f"Connection test successful. User count: {user_count}")
            except Exception as test_error:
                logger.warning(f"Connection test query failed: {test_error}")
                # Continue anyway since the client was created successfully

            return True
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            logger.error(traceback.format_exc())
            return False

    def register_user(self, username: str, email: Optional[str] = None, display_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Register a new user.

        Args:
            username: Unique username for the user
            email: Optional email address
            display_name: Optional display name

        Returns:
            User data dictionary if successful, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return None

        try:
            # Check if username already exists
            existing_user = self.get_user_by_username(username)
            if existing_user:
                logger.error(f"Username '{username}' already exists")
                return None

            # Create user data
            user_id = str(uuid.uuid4())
            user_data = {
                'id': user_id,
                'username': username,
                'email': email,
                'display_name': display_name or username,
                'created_at': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat(),
                'preferences': {}
            }

            # Insert user into database
            self.supabase_client.table('users').insert(user_data).execute()
            logger.info(f"User registered successfully: {username}")

            # Set as current user
            self.current_user = user_data

            return user_data
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return None

    def login_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Log in a user by username.

        Args:
            username: Username to log in

        Returns:
            User data dictionary if successful, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return None

        try:
            # Get user by username
            user = self.get_user_by_username(username)
            if not user:
                logger.error(f"User not found: {username}")
                return None

            # Update last login time
            self.supabase_client.table('users').update({
                'last_login': datetime.now().isoformat()
            }).eq('id', user['id']).execute()

            # Set as current user
            self.current_user = user
            logger.info(f"User logged in: {username}")

            return user
        except Exception as e:
            logger.error(f"Error logging in user: {e}")
            return None

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user data by username.

        Args:
            username: Username to look up

        Returns:
            User data dictionary if found, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return None

        try:
            response = self.supabase_client.table('users').select('*').eq('username', username).execute()

            if not response.data or len(response.data) == 0:
                return None

            return response.data[0]
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID.

        Args:
            user_id: User ID to look up

        Returns:
            User data dictionary if found, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return None

        try:
            response = self.supabase_client.table('users').select('*').eq('id', user_id).execute()

            if not response.data or len(response.data) == 0:
                return None

            return response.data[0]
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user profile information.

        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            # Ensure we're not updating the ID
            if 'id' in updates:
                del updates['id']

            # Update user
            self.supabase_client.table('users').update(updates).eq('id', user_id).execute()
            logger.info(f"User profile updated: {user_id}")

            # Update current user if this is the current user
            if self.current_user and self.current_user.get('id') == user_id:
                self.current_user.update(updates)

            return True
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def list_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all users.

        Args:
            limit: Maximum number of users to return

        Returns:
            List of user data dictionaries
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return []

        try:
            response = self.supabase_client.table('users').select('*').limit(limit).execute()

            if not response.data:
                logger.info("No users found")
                return []

            logger.info(f"Retrieved {len(response.data)} users")
            return response.data
        except Exception as e:
            logger.error(f"Error listing users: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """Delete a user and all associated memories.

        Args:
            user_id: ID of the user to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            # First delete all memories associated with this user
            self.supabase_client.table('memories').delete().eq('user_id', user_id).execute()

            # Then delete the user
            self.supabase_client.table('users').delete().eq('id', user_id).execute()

            # Clear current user if this was the current user
            if self.current_user and self.current_user.get('id') == user_id:
                self.current_user = None

            logger.info(f"User deleted: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return False

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get the currently logged in user.

        Returns:
            Current user data dictionary if logged in, None otherwise
        """
        return self.current_user

    def logout(self) -> bool:
        """Log out the current user.

        Returns:
            True if successful, False otherwise
        """
        if not self.current_user:
            logger.warning("No user is currently logged in")
            return False

        logger.info(f"User logged out: {self.current_user.get('username')}")
        self.current_user = None
        return True


# Example usage
def test_user_account_manager():
    """Test the user account manager."""
    # Initialize user account manager
    manager = UserAccountManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return False

    # Register a test user
    test_username = f"test_user_{int(time.time())}"
    user = manager.register_user(
        username=test_username,
        email=f"{test_username}@example.com",
        display_name="Test User"
    )

    if not user:
        logger.error("Failed to register test user")
        return False

    logger.info(f"Registered test user: {user}")

    # Test login
    logged_in_user = manager.login_user(test_username)
    if not logged_in_user:
        logger.error("Failed to log in test user")
        return False

    logger.info(f"Logged in test user: {logged_in_user}")

    # Test update profile
    update_success = manager.update_user_profile(
        user_id=user['id'],
        updates={'display_name': 'Updated Test User'}
    )

    if not update_success:
        logger.error("Failed to update test user profile")
    else:
        logger.info("Updated test user profile")

    # Test get current user
    current_user = manager.get_current_user()
    if current_user:
        logger.info(f"Current user: {current_user}")

    # Test logout
    logout_success = manager.logout()
    if logout_success:
        logger.info("Logged out test user")

    # Clean up - delete test user
    delete_success = manager.delete_user(user['id'])
    if delete_success:
        logger.info("Deleted test user")
    else:
        logger.error("Failed to delete test user")

    return True

if __name__ == "__main__":
    test_user_account_manager()
