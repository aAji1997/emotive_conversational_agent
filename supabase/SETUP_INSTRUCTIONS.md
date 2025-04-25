# Supabase Database Setup Instructions

This document provides detailed instructions for setting up the Supabase database for the memory system. These instructions are designed to ensure a consistent setup across all collaborators.

## Automatic Setup (Recommended)

The application now includes an automatic database initialization process that runs when you start the application. This should handle all the necessary setup for you.

1. Make sure you have a valid `.api_key.json` file with your Supabase credentials:
   ```json
   {
     "openai_api_key": "your-openai-api-key",
     "gemini_api_key": "your-gemini-api-key",
     "supabase_url": "your-supabase-url",
     "supabase_api_key": "your-supabase-api-key"
   }
   ```

2. Run the application normally:
   ```
   python gpt_realtime/realtime_audio_gpt.py
   ```

3. The application will automatically initialize the database with the correct schema and functions.

## Manual Setup

If you encounter issues with the automatic setup, you can run the initialization script manually:

```
python supabase/initialize_database.py
```

This script will:
1. Create the necessary tables if they don't exist
2. Create the required SQL functions
3. Handle any schema migrations (e.g., if you have a `created_at` column but need a `timestamp` column)
4. Verify that everything is set up correctly

## Verifying Your Setup

You can verify that your database is set up correctly by running:

```
python supabase/verify_database.py
```

This will check:
- If all required tables exist
- If all required columns exist in the tables
- If all required functions exist and have the correct signatures
- If the functions can be executed successfully

If any issues are found, the script will provide recommendations for fixing them.

## Common Issues and Solutions

### "column m.created_at does not exist" Error

This error occurs when the SQL functions are trying to access a column that doesn't exist in your database. The solution is to run the database initialization script, which will update the functions to use the correct column names.

```
python supabase/initialize_database.py
```

### Schema Inconsistencies

If you're experiencing issues due to schema inconsistencies (e.g., some collaborators have a `created_at` column while others have a `timestamp` column), the initialization script includes a migration function that will handle these differences.

### Manual SQL Execution

If you need to manually execute the SQL setup script, you can do so through the Supabase dashboard:

1. Go to the [Supabase Dashboard](https://app.supabase.com/)
2. Select your project
3. Click on "SQL Editor" in the left sidebar
4. Click "New Query" to create a new SQL query
5. Copy and paste the contents of `supabase/standard_schema.sql`
6. Click "Run" to execute the SQL

## Database Schema

The memory system uses the following tables:

### `users` Table
- `id`: TEXT PRIMARY KEY
- `username`: TEXT UNIQUE NOT NULL
- `email`: TEXT UNIQUE
- `display_name`: TEXT
- `created_at`: TIMESTAMPTZ NOT NULL DEFAULT NOW()
- `last_login`: TIMESTAMPTZ
- `preferences`: JSONB DEFAULT '{}'::JSONB

### `memories` Table
- `id`: TEXT PRIMARY KEY
- `content`: TEXT NOT NULL
- `embedding`: VECTOR(300)
- `importance`: INTEGER CHECK (importance >= 1 AND importance <= 10)
- `category`: TEXT NOT NULL
- `timestamp`: TIMESTAMPTZ NOT NULL
- `source`: TEXT
- `conversation_mode`: TEXT
- `user_id`: TEXT REFERENCES users(id)
- `metadata`: JSONB DEFAULT '{}'::JSONB

## SQL Functions

The memory system uses the following SQL functions:

### `match_memories`
Matches memories based on vector similarity.

### `match_memories_by_user`
Matches memories based on vector similarity and user ID.

### `clear_all_memories`
Clears all memories from the database (for testing/reset purposes).

### `check_and_migrate_schema`
Handles schema migrations between different versions of the database.

## Support

If you continue to experience issues with the database setup, please contact the project maintainers for assistance.
