-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the memories table
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

-- Create an index on the embedding column for faster similarity searches
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create a function to match memories based on embedding similarity
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

-- Create a function to clear all memories (for testing/reset purposes)
CREATE OR REPLACE FUNCTION clear_all_memories()
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM memories;
END;
$$;
