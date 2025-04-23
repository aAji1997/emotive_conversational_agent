import os
import sys
import logging
import argparse
from pathlib import Path
from user_account_manager import UserAccountManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function for user registration and login CLI."""
    parser = argparse.ArgumentParser(description="User Account Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a new user")
    register_parser.add_argument("--username", required=True, help="Username for the new user")
    register_parser.add_argument("--email", help="Email address (optional)")
    register_parser.add_argument("--display-name", help="Display name (optional)")
    
    # Login command
    login_parser = subparsers.add_parser("login", help="Log in as an existing user")
    login_parser.add_argument("--username", required=True, help="Username to log in with")
    
    # List users command
    list_parser = subparsers.add_parser("list", help="List all users")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of users to list")
    
    # Update profile command
    update_parser = subparsers.add_parser("update", help="Update user profile")
    update_parser.add_argument("--username", required=True, help="Username of the user to update")
    update_parser.add_argument("--email", help="New email address")
    update_parser.add_argument("--display-name", help="New display name")
    
    # Delete user command
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("--username", required=True, help="Username of the user to delete")
    
    args = parser.parse_args()
    
    # Initialize user account manager
    manager = UserAccountManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return
    
    # Execute command
    if args.command == "register":
        user = manager.register_user(
            username=args.username,
            email=args.email,
            display_name=args.display_name
        )
        
        if user:
            print(f"\nUser registered successfully:")
            print(f"Username: {user.get('username')}")
            print(f"Display Name: {user.get('display_name')}")
            print(f"User ID: {user.get('id')}")
        else:
            print("\nFailed to register user. Username may already exist.")
    
    elif args.command == "login":
        user = manager.login_user(args.username)
        
        if user:
            print(f"\nLogged in successfully:")
            print(f"Username: {user.get('username')}")
            print(f"Display Name: {user.get('display_name')}")
            print(f"User ID: {user.get('id')}")
            print(f"Last Login: {user.get('last_login')}")
        else:
            print(f"\nFailed to log in. User '{args.username}' not found.")
    
    elif args.command == "list":
        users = manager.list_users(args.limit)
        
        if users:
            print(f"\nUsers ({len(users)}):")
            for i, user in enumerate(users):
                print(f"{i+1}. {user.get('username')} - {user.get('display_name')} (ID: {user.get('id')})")
        else:
            print("\nNo users found.")
    
    elif args.command == "update":
        # First get the user to update
        user = manager.get_user_by_username(args.username)
        
        if not user:
            print(f"\nUser '{args.username}' not found.")
            return
        
        # Prepare updates
        updates = {}
        if args.email:
            updates['email'] = args.email
        if args.display_name:
            updates['display_name'] = args.display_name
        
        if not updates:
            print("\nNo updates specified.")
            return
        
        # Update the user
        success = manager.update_user_profile(user['id'], updates)
        
        if success:
            print(f"\nUser '{args.username}' updated successfully.")
        else:
            print(f"\nFailed to update user '{args.username}'.")
    
    elif args.command == "delete":
        # First get the user to delete
        user = manager.get_user_by_username(args.username)
        
        if not user:
            print(f"\nUser '{args.username}' not found.")
            return
        
        # Confirm deletion
        confirm = input(f"Are you sure you want to delete user '{args.username}'? This will also delete all associated memories. (y/n): ")
        
        if confirm.lower() != 'y':
            print("\nDeletion cancelled.")
            return
        
        # Delete the user
        success = manager.delete_user(user['id'])
        
        if success:
            print(f"\nUser '{args.username}' deleted successfully.")
        else:
            print(f"\nFailed to delete user '{args.username}'.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
