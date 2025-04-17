# Memory Storage System

This document explains how to set up and use the memory storage system for the emotive conversational agent.

## Overview

The memory storage system uses PostgreSQL with the pgvector extension to store and retrieve memories as vector embeddings. This allows for semantic similarity search of memories based on their content.

## Setup

### Prerequisites

1. PostgreSQL database with pgvector extension enabled
2. Python 3.8+ with the following packages:
   - psycopg2
   - spacy (with en_core_web_md model)
   - python-dotenv

### Configuration

1. Create a `.env` file in the project root with your PostgreSQL credentials:

```
user=your_postgres_user
password=your_postgres_password
host=your_postgres_host
port=your_postgres_port
dbname=your_postgres_database
```

2. Run the setup script to create the necessary tables and functions:

```bash
python supabase/setup_memory_storage.py
```

## Usage

### Command-line Utilities

The `memory_utils_direct.py` script provides command-line utilities for managing memories:

```bash
# List memories
python memory_utils_direct.py list

# Add a memory
python memory_utils_direct.py add --content "User enjoys hiking" --importance 8 --category "preference"

# Search memories
python memory_utils_direct.py search --query "What activities does the user enjoy?"

# Delete a memory
python memory_utils_direct.py delete --id "memory-id"

# Clear all memories
python memory_utils_direct.py clear
```

### Integration with the Conversational Agent

The `memory/memory_agent_direct.py` file contains the `ConversationMemoryAgent` class that integrates with the conversational agent. It provides the following functionality:

- Storing memories from conversations
- Retrieving relevant memories based on the current conversation
- Formatting memories for inclusion in the conversation context

To use the memory agent in your conversational agent:

```python
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from memory.memory_agent_direct import ConversationMemoryAgent

# Initialize models
storage_model = LiteLlm(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.2
)

retrieval_model = LiteLlm(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.3
)

orchestrator_model = LiteLlm(
    model="gpt-4o",
    api_key=openai_api_key,
    temperature=0.1
)

# Initialize session service
session_service = InMemorySessionService()

# Initialize memory agent
memory_agent = ConversationMemoryAgent(
    storage_agent_model=storage_model,
    retrieval_agent_model=retrieval_model,
    orchestrator_agent_model=orchestrator_model,
    session_service=session_service
)

# Process a message
await memory_agent.process_message(
    message="I enjoy hiking in the mountains on weekends.",
    role="user",
    conversation_mode="chat"
)

# Get relevant memories for the current message
relevant_memories = await memory_agent.get_relevant_memories(
    "What outdoor activities do you recommend?"
)

# Format memories for inclusion in the conversation context
formatted_memories = memory_agent.format_memories_for_context(relevant_memories)
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
  query_embedding => ARRAY[0.1, 0.2, ...],  -- Float array of the query embedding
  match_threshold => 0.3,                   -- Similarity threshold (0-1)
  match_count => 5                          -- Maximum number of results
);
```

## Troubleshooting

If you encounter issues with the memory system:

1. Check that the pgvector extension is enabled in your PostgreSQL database
2. Verify that your `.env` file contains the correct database credentials
3. Run the `test_direct_connection.py` script to test the database connection
4. Check the logs for detailed error messages

## Files

- `supabase/setup_memory_storage.py`: Script to set up the memory storage system
- `memory_utils_direct.py`: Command-line utilities for managing memories
- `memory/memory_agent_direct.py`: Memory agent for integration with the conversational agent
- `test_direct_connection.py`: Script to test the database connection
