-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for users if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    preferences JSONB DEFAULT '{}'::JSONB
);

-- Create a table for memories
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(300),
    importance INTEGER CHECK (importance >= 1 AND importance <= 10),
    category TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    source TEXT,
    conversation_mode TEXT,
    user_id TEXT REFERENCES users(id),
    metadata JSONB DEFAULT '{}'::JSONB
);

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create an index on user_id for faster queries
CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories(user_id);

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS match_memories(VECTOR(300), FLOAT, INTEGER);
DROP FUNCTION IF EXISTS match_memories_by_user(VECTOR(300), FLOAT, INTEGER, TEXT);

-- Create a function to match memories by vector similarity
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR(300),
    match_threshold FLOAT,
    match_count INTEGER
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    embedding TEXT,
    similarity FLOAT,
    importance INTEGER,
    category TEXT,
    source TEXT,
    user_id TEXT,
    memory_timestamp TIMESTAMPTZ,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.embedding::text as embedding,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.importance,
        m.category,
        m.source,
        m.user_id,
        m.timestamp as memory_timestamp,
        m.metadata
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

-- Create a function to match memories by vector similarity and user_id
CREATE OR REPLACE FUNCTION match_memories_by_user(
    query_embedding VECTOR(300),
    match_threshold FLOAT,
    match_count INTEGER,
    user_id_param TEXT
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    embedding TEXT,
    similarity FLOAT,
    importance INTEGER,
    category TEXT,
    source TEXT,
    user_id TEXT,
    memory_timestamp TIMESTAMPTZ,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.embedding::text as embedding,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.importance,
        m.category,
        m.source,
        m.user_id,
        m.timestamp as memory_timestamp,
        m.metadata
    FROM
        memories m
    WHERE
        1 - (m.embedding <=> query_embedding) > match_threshold
        AND (m.user_id = user_id_param OR m.user_id IS NULL)
    ORDER BY
        m.embedding <=> query_embedding
    LIMIT
        match_count;
END;
$$;

-- Create a function to clear all memories (for testing/reset purposes)
CREATE OR REPLACE FUNCTION clear_all_memories()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM memories;
END;
$$;

-- Create a function to check if created_at column exists and migrate if needed
CREATE OR REPLACE FUNCTION check_and_migrate_schema()
RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    migration_message TEXT := 'No migration needed';
BEGIN
    -- Check if created_at column exists in memories table
    IF EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'memories' AND column_name = 'created_at'
    ) THEN
        -- If timestamp column doesn't exist, add it
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = 'memories' AND column_name = 'timestamp'
        ) THEN
            ALTER TABLE memories ADD COLUMN timestamp TIMESTAMPTZ;
            UPDATE memories SET timestamp = created_at;
            migration_message := 'Added timestamp column and copied data from created_at';
        ELSE
            -- Both columns exist, ensure timestamp has data
            UPDATE memories SET timestamp = created_at WHERE timestamp IS NULL;
            migration_message := 'Ensured timestamp column has data from created_at';
        END IF;
    END IF;
    
    RETURN migration_message;
END;
$$;
