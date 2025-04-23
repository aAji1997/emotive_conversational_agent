# Memory Storage System

This directory contains the setup files for the Supabase memory storage system used by the emotive conversational agent.

## Overview

The memory storage system uses Supabase with the pgvector extension to store and retrieve memories as vector embeddings. This allows for semantic similarity search of memories based on their content.

## Files

- `setup.sql`: SQL script to set up the necessary tables, functions, and indexes in Supabase
- `init_database.py`: Python script to initialize the database
- `execute_setup.py`: Alternative script to execute the SQL setup directly via the REST API

## Setup Instructions

1. Ensure your Supabase credentials are in the `.api_key.json` file in the project root:
   ```json
   {
     "supabase_url": "YOUR_SUPABASE_URL",
     "supabase_api_key": "YOUR_SUPABASE_API_KEY"
   }
   ```

2. Make sure the pgvector extension is enabled in your Supabase project:
   - Go to the Supabase dashboard
   - Navigate to the SQL Editor
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`

3. Run the initialization script:
   ```bash
   python initialize_memory_system.py
   ```

4. Verify the setup by using the memory utilities:
   ```bash
   python memory_utils.py list
   ```

## Database Schema

The `memories` table has the following structure:

- `id`: Text primary key
- `content`: Text content of the memory
- `embedding`: Vector(300) representation of the content
- `importance`: Integer (1-10) indicating the importance of the memory
- `category`: Text category of the memory (e.g., preference, personal_info)
- `timestamp`: Timestamp when the memory was created
- `source`: Text source of the memory (e.g., user, assistant)
- `conversation_mode`: Text mode of the conversation (e.g., audio, chat)
- `metadata`: JSONB for additional metadata

## Vector Similarity Search

The `match_memories` function is used to find memories similar to a query embedding:

```sql
SELECT * FROM match_memories(
  query_embedding => [0.1, 0.2, ...],  -- Vector embedding of the query
  match_threshold => 0.5,              -- Similarity threshold (0-1)
  match_count => 5                     -- Maximum number of results
);
```

## Memory Management

Use the `memory_utils.py` script to manage memories:

```bash
# List memories
python memory_utils.py list

# Add a memory
python memory_utils.py add --content "User likes hiking" --importance 7 --category "preference"

# Search memories
python memory_utils.py search --query "What activities does the user enjoy?"

# Delete a memory
python memory_utils.py delete --id "memory-id"

# Clear all memories
python memory_utils.py clear
```
