-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a table for users if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a table for memories
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(300) NOT NULL,
    importance INTEGER CHECK (importance >= 1 AND importance <= 10),
    category TEXT,
    source TEXT,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::JSONB
);

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Drop existing functions if they exist
DROP FUNCTION IF EXISTS match_memories(VECTOR(300), FLOAT, INTEGER);
DROP FUNCTION IF EXISTS match_memories_by_user(VECTOR(300), FLOAT, INTEGER, UUID);

-- Create a function to match memories by vector similarity
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR(300),
    match_threshold FLOAT,
    match_count INTEGER
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    similarity FLOAT,
    importance INTEGER,
    category TEXT,
    source TEXT,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.importance,
        m.category,
        m.source,
        m.user_id,
        m.created_at,
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
    user_id_param UUID
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    similarity FLOAT,
    importance INTEGER,
    category TEXT,
    source TEXT,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        1 - (m.embedding <=> query_embedding) as similarity,
        m.importance,
        m.category,
        m.source,
        m.user_id,
        m.created_at,
        m.metadata
    FROM
        memories m
    WHERE
        1 - (m.embedding <=> query_embedding) > match_threshold
        AND m.user_id = user_id_param
    ORDER BY
        m.embedding <=> query_embedding
    LIMIT
        match_count;
END;
$$;
