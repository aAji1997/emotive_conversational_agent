# User Account System

This document explains how to set up and use the user account system for the emotive conversational agent.

## Overview

The user account system allows users to create and maintain accounts, which enables personalization and user-specific memory retrieval. This system integrates with the existing memory storage system to associate memories with specific users.

## Setup

### Prerequisites

1. Supabase project with the memory storage system already set up
2. Python 3.8+ with the following packages:
   - supabase
   - spacy (with en_core_web_md model)

### Configuration

1. Ensure your Supabase credentials are in the `.api_key.json` file in the project root:
   ```json
   {
     "supabase_url": "YOUR_SUPABASE_URL",
     "supabase_api_key": "YOUR_SUPABASE_API_KEY"
   }
   ```

2. Run the initialization script to set up the user account system:
   ```bash
   python initialize_user_system.py
   ```

## Usage

### User Account Management

The `user_registration.py` script provides command-line utilities for managing user accounts:

```bash
# Register a new user
python user_registration.py register --username "john_doe" --email "john@example.com" --display-name "John Doe"

# Log in as a user
python user_registration.py login --username "john_doe"

# List all users
python user_registration.py list

# Update a user's profile
python user_registration.py update --username "john_doe" --display-name "John D."

# Delete a user (this will also delete all associated memories)
python user_registration.py delete --username "john_doe"
```

### User-Specific Memory Management

The `user_memory_utils.py` script provides command-line utilities for managing user-specific memories:

```bash
# List memories for a user
python user_memory_utils.py list --username "john_doe"

# Add a memory for a user
python user_memory_utils.py add --username "john_doe" --content "John enjoys hiking on weekends" --importance 8 --category "preference"

# Search memories for a user
python user_memory_utils.py search --username "john_doe" --query "What does the user enjoy doing?"

# Delete a memory for a user
python user_memory_utils.py delete --username "john_doe" --memory-id "memory-id"

# Clear all memories for a user
python user_memory_utils.py clear --username "john_doe"
```

## Integration with Memory Manager

The memory manager has been updated to support user-specific memories. When adding or retrieving memories, you can now specify a `user_id` parameter to associate memories with a specific user:

```python
from user_account_manager import UserAccountManager
from memory_manager_improved import ImprovedMemoryManager

# Initialize managers
user_manager = UserAccountManager()
memory_manager = ImprovedMemoryManager()

# Connect to Supabase
user_manager.connect()
memory_manager.connect()

# Register a user
user = user_manager.register_user(username="john_doe", display_name="John Doe")
user_id = user["id"]

# Add a memory for the user
memory_manager.add_memory(
    content="John enjoys hiking on weekends",
    importance=8,
    category="preference",
    user_id=user_id
)

# List memories for the user
user_memories = memory_manager.list_memories(user_id=user_id)

# Search memories for the user
search_results = memory_manager.search_memories(
    query="What does the user enjoy doing?",
    user_id=user_id
)
```

## Database Schema

The user account system adds the following to the database schema:

1. `users` table:
   - `id`: Text primary key
   - `username`: Text unique identifier
   - `email`: Text (optional)
   - `display_name`: Text
   - `created_at`: Timestamp when the user was created
   - `last_login`: Timestamp of the last login
   - `preferences`: JSONB for additional user preferences

2. `user_id` column added to the `memories` table to associate memories with users

## Testing

To test the user account system, run:

```bash
python test_user_memory_system.py
```

This script will:
1. Create test users
2. Add memories for each user
3. Test listing and searching memories for each user
4. Clean up by deleting the test data

## Troubleshooting

If you encounter issues with the user account system:

1. Check that the Supabase credentials in `.api_key.json` are correct
2. Verify that the user account system was initialized successfully
3. Check the logs for detailed error messages
4. Ensure that the memory storage system is set up correctly
