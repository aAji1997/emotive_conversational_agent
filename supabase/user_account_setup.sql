-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    preferences JSONB DEFAULT '{}'::JSONB
);

-- Add user_id column to memories table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = 'memories' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE memories ADD COLUMN user_id TEXT REFERENCES users(id);
        
        -- Create an index on user_id for faster queries
        CREATE INDEX IF NOT EXISTS memories_user_id_idx ON memories(user_id);
    END IF;
END $$;

-- Create function to get memories for a specific user
CREATE OR REPLACE FUNCTION get_user_memories(
    user_id_param TEXT,
    limit_param INT DEFAULT 100
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
    user_id TEXT
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
        m.user_id
    FROM
        memories m
    WHERE
        m.user_id = user_id_param
    ORDER BY
        m.timestamp DESC
    LIMIT
        limit_param;
END;
$$;

-- Modify match_memories function to filter by user_id
CREATE OR REPLACE FUNCTION match_memories(
    query_embedding VECTOR(300),
    match_threshold FLOAT,
    match_count INT,
    user_id_param TEXT DEFAULT NULL
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
    similarity FLOAT,
    user_id TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF user_id_param IS NULL THEN
        RETURN QUERY
        SELECT
            m.id,
            m.content,
            m.importance,
            m.category,
            m.timestamp,  -- Changed from created_at to timestamp
            m.source,
            m.conversation_mode,
            m.metadata,
            1 - (m.embedding <=> query_embedding) AS similarity,
            m.user_id
        FROM
            memories m
        WHERE
            1 - (m.embedding <=> query_embedding) > match_threshold
        ORDER BY
            m.embedding <=> query_embedding
        LIMIT
            match_count;
    ELSE
        RETURN QUERY
        SELECT
            m.id,
            m.content,
            m.importance,
            m.category,
            m.timestamp,  -- Changed from created_at to timestamp
            m.source,
            m.conversation_mode,
            m.metadata,
            1 - (m.embedding <=> query_embedding) AS similarity,
            m.user_id
        FROM
            memories m
        WHERE
            1 - (m.embedding <=> query_embedding) > match_threshold
            AND m.user_id = user_id_param
        ORDER BY
            m.embedding <=> query_embedding
        LIMIT
            match_count;
    END IF;
END;
$$;

