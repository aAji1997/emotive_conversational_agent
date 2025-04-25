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
    memory_timestamp TIMESTAMP WITH TIME ZONE,
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
    memory_timestamp TIMESTAMP WITH TIME ZONE,
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
