# Supabase Memory System Setup Instructions

The initialization script encountered some issues with executing SQL statements directly through the API. Follow these manual steps to set up your Supabase database for the memory system.

## Step 1: Access the Supabase SQL Editor

1. Go to the [Supabase Dashboard](https://app.supabase.com/)
2. Select your project
3. Click on "SQL Editor" in the left sidebar
4. Click "New Query" to create a new SQL query

## Step 2: Enable the pgvector Extension

Run the following SQL command:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Step 3: Create the Memories Table

Run the following SQL command:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(300), -- Using 300 dimensions for spaCy's en_core_web_md model
    importance INTEGER CHECK (importance >= 1 AND importance <= 10),
    category TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source TEXT,
    conversation_mode TEXT,
    metadata JSONB DEFAULT '{}'::JSONB
);
```

## Step 4: Create an Index for Vector Similarity Search

Run the following SQL command:

```sql
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

## Step 5: Create the Match Memories Function

Run the following SQL command:

```sql
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR(300),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    importance INTEGER,
    category TEXT,
    timestamp TIMESTAMPTZ,
    source TEXT,
    conversation_mode TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.importance,
        m.category,
        m.timestamp,
        m.source,
        m.conversation_mode,
        m.metadata,
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM
        memories m
    WHERE
        1 - (m.embedding <=> query_embedding) > match_threshold
    ORDER BY
        m.embedding <=> query_embedding
    LIMIT
        match_count;
END;
$$;
```

## Step 6: Create a Function to Clear All Memories

Run the following SQL command:

```sql
CREATE OR REPLACE FUNCTION clear_all_memories()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM memories;
END;
$$;
```

## Step 7: Verify the Setup

After executing all the SQL commands, you can verify the setup by running:

```sql
SELECT * FROM memories LIMIT 10;
```

This should return an empty result set (since no memories have been added yet), but without any errors.

## Step 8: Test the Memory System

Run the memory initialization script again to test the system:

```bash
python initialize_memory_system.py
```

Or use the memory utilities to add and retrieve memories:

```bash
python memory_utils.py add --content "This is a test memory" --importance 5 --category "test"
python memory_utils.py list
```

## Troubleshooting

If you encounter any issues:

1. Check that the pgvector extension is enabled
2. Verify that all tables and functions were created successfully
3. Ensure your Supabase API key has the necessary permissions
4. Check the Supabase logs for any error messages
