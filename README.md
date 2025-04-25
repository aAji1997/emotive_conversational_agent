# Emotive Conversational Agent

A conversational agent with emotional intelligence using Google's Gemini API and OpenAI's GPT models.

## Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/aAji1997/emotive_conversational_agent.git
   cd emotive_conversational_agent
   ```

2. Create a `.api_key.json` file with your API keys:
   ```json
   {
     "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
     "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
     "supabase_url": "YOUR_SUPABASE_URL",
     "supabase_api_key": "YOUR_SUPABASE_API_KEY"
   }
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python gpt_realtime/realtime_audio_gpt.py
   ```

The application will automatically set up the database and memory system on first run.

## Detailed Setup

### API Keys

You'll need API keys from:
- OpenAI (for GPT-4o and real-time audio)
- Google (for Gemini models)
- Supabase (for memory storage)

### Database Setup

The application now includes automatic database initialization. When you run the application for the first time, it will:
1. Create necessary tables in your Supabase database
2. Set up required SQL functions
3. Handle any schema migrations automatically

If you prefer manual setup or encounter issues, you can:
1. Run the database initialization script directly:
   ```bash
   python supabase/initialize_database.py
   ```
2. Verify your database setup:
   ```bash
   python supabase/verify_database.py
   ```

### Optional: Direct Database Connection

For advanced features, you can create a `.env` file with direct PostgreSQL connection details:
```
user=your_postgres_user
password=your_postgres_password
host=your_postgres_host
port=your_postgres_port
dbname=your_postgres_database
```
This is optional for most users as the application primarily uses the Supabase API.

## Features

- AI-powered conversations using OpenAI's GPT and Google's Gemini models
- Emotion detection and realtime emotion analysis
- Natural language understanding

## Technologies

- Python
- OpenAI real-time API for real-time audio conversations
- Google Gemini API and Google ADK for sentiment analysis and report generation

## Running the Application

### Audio Conversation Mode (Recommended)

The primary application uses OpenAI's real-time audio API for natural voice conversations:

```bash
python gpt_realtime/realtime_audio_gpt.py
```

This will:
- Automatically set up the database
- Prompt you for a username
- Start a voice conversation interface
- Enable real-time sentiment analysis

### Command Line Arguments

The application supports several command line arguments:

```bash
python gpt_realtime/realtime_audio_gpt.py --chat-mode --no-memory --no-sentiment
```

- `--chat-mode`: Use text chat instead of voice
- `--no-memory`: Disable the memory system
- `--no-sentiment`: Disable sentiment analysis

### Requirements

- OpenAI API key with access to GPT-4o and real-time audio API
- Microphone and speakers for audio mode
- Supabase account for memory features

## Memory System

This application includes a memory system that stores and retrieves user interactions. The memory system uses Supabase with the pgvector extension for semantic similarity search.

### Memory System Features

- **Long-term Memory**: The agent remembers details from previous conversations
- **Semantic Search**: Retrieves memories based on semantic similarity, not just keyword matching
- **User-specific Memories**: Memories are associated with specific users for personalized experiences
- **Automatic Deduplication**: Similar memories are automatically consolidated to prevent redundancy

### Memory System Setup

The memory system is now **automatically initialized** when you run the application. All you need is:

- Supabase credentials in your `.api_key.json` file

The direct PostgreSQL connection via `.env` is now optional for most users.

### Troubleshooting

If you encounter any database-related issues:

1. Run the verification tool to diagnose problems:
   ```bash
   python supabase/verify_database.py
   ```

2. Run the initialization script to fix common issues:
   ```bash
   python supabase/initialize_database.py
   ```

3. For detailed instructions, see `supabase/SETUP_INSTRUCTIONS.md`

### User Accounts

The application supports user accounts, which allows for personalized memory retrieval. When you start the application, you'll be prompted to enter a username. This username is used to associate memories with specific users.