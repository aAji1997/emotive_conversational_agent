import os
import sys
import logging
import argparse
from pathlib import Path
from accounts.user_account_manager import UserAccountManager
from memory_manager_improved import ImprovedMemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function for user-specific memory management CLI."""
    parser = argparse.ArgumentParser(description="User Memory Management")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List memories command
    list_parser = subparsers.add_parser("list", help="List memories for a user")
    list_parser.add_argument("--username", required=True, help="Username to list memories for")
    list_parser.add_argument("--limit", type=int, default=10, help="Maximum number of memories to list")
    
    # Add memory command
    add_parser = subparsers.add_parser("add", help="Add a memory for a user")
    add_parser.add_argument("--username", required=True, help="Username to add memory for")
    add_parser.add_argument("--content", required=True, help="Content of the memory")
    add_parser.add_argument("--importance", type=int, required=True, help="Importance of the memory (1-10)")
    add_parser.add_argument("--category", required=True, help="Category of the memory")
    add_parser.add_argument("--source", default="manual", help="Source of the memory")
    add_parser.add_argument("--conversation-mode", default="manual", help="Conversation mode")
    
    # Search memories command
    search_parser = subparsers.add_parser("search", help="Search memories for a user")
    search_parser.add_argument("--username", required=True, help="Username to search memories for")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    # Delete memory command
    delete_parser = subparsers.add_parser("delete", help="Delete a memory for a user")
    delete_parser.add_argument("--username", required=True, help="Username to delete memory for")
    delete_parser.add_argument("--memory-id", required=True, help="ID of the memory to delete")
    
    # Clear all memories command
    clear_parser = subparsers.add_parser("clear", help="Clear all memories for a user")
    clear_parser.add_argument("--username", required=True, help="Username to clear memories for")
    
    args = parser.parse_args()
    
    # Initialize managers
    user_manager = UserAccountManager()
    if not user_manager.connect():
        logger.error("Failed to connect to Supabase (user manager)")
        return
    
    memory_manager = ImprovedMemoryManager()
    if not memory_manager.connect():
        logger.error("Failed to connect to Supabase (memory manager)")
        return
    
    # Get user by username
    if args.command:
        user = user_manager.get_user_by_username(args.username)
        if not user:
            print(f"\nUser '{args.username}' not found.")
            return
        
        user_id = user['id']
    
    # Execute command
    if args.command == "list":
        memories = memory_manager.list_memories(args.limit, user_id=user_id)
        
        if memories:
            print(f"\nMemories for user '{args.username}' ({len(memories)}):")
            for i, memory in enumerate(memories):
                print(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Importance: {memory.get('importance', 0)})")
        else:
            print(f"\nNo memories found for user '{args.username}'.")
    
    elif args.command == "add":
        memory_id = memory_manager.add_memory(
            content=args.content,
            importance=args.importance,
            category=args.category,
            source=args.source,
            conversation_mode=args.conversation_mode,
            user_id=user_id
        )
        
        if memory_id:
            print(f"\nMemory added successfully for user '{args.username}' with ID: {memory_id}")
        else:
            print(f"\nFailed to add memory for user '{args.username}'.")
    
    elif args.command == "search":
        results = memory_manager.search_memories(
            query=args.query,
            limit=args.limit,
            user_id=user_id
        )
        
        if results:
            print(f"\nSearch results for user '{args.username}' ({len(results)}):")
            for i, memory in enumerate(results):
                similarity = memory.get('combined_score', memory.get('similarity', 0))
                print(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            print(f"\nNo relevant memories found for user '{args.username}'.")
    
    elif args.command == "delete":
        success = memory_manager.delete_memory(args.memory_id)
        
        if success:
            print(f"\nMemory deleted successfully for user '{args.username}'.")
        else:
            print(f"\nFailed to delete memory for user '{args.username}'.")
    
    elif args.command == "clear":
        confirm = input(f"Are you sure you want to clear all memories for user '{args.username}'? This cannot be undone. (y/n): ")
        
        if confirm.lower() != 'y':
            print("\nOperation cancelled.")
            return
        
        # We need to get all memories for this user and delete them one by one
        # since clear_all_memories() would clear ALL memories in the database
        memories = memory_manager.list_memories(limit=1000, user_id=user_id)
        
        if not memories:
            print(f"\nNo memories found for user '{args.username}'.")
            return
        
        deleted_count = 0
        for memory in memories:
            if memory_manager.delete_memory(memory['id']):
                deleted_count += 1
        
        print(f"\nDeleted {deleted_count} memories for user '{args.username}'.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
