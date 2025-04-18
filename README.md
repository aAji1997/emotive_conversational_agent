# Emotive Conversational Agent

A conversational agent with emotional intelligence using Google's Gemini API.

## Setup

1. Clone this repository
2. Create a `.api_key.json` file with your API keys:
```json
{
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
  "supabase_url": "YOUR_SUPABASE_URL",
  "supabase_api_key": "YOUR_SUPABASE_API_KEY"
}
```
3. Create a `.env` file in the project root with your database credentials (you can copy and modify the provided `.env.example` file):
```
user=your_postgres_user
password=your_postgres_password
host=your_postgres_host
port=your_postgres_port
dbname=your_postgres_database
```
   This file is required for the memory system to establish direct database connections with PostgreSQL.
4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- AI-powered conversations using OpenAI's GPT and Google's Gemini models
- Emotion detection and realtime emotion analysis
- Natural language understanding

## Technologies

- Python
- OpenAI real-time API for real-time audio conversations
- Google Gemini API and Google ADK for sentiment analysis and report generation

## Running the Application

The `realtime_audio_gpt.py` script is more stable than `gemini_live_audio.py`. You can run it directly without any additional setup:

```bash
python gpt_realtime/realtime_audio_gpt.py
```

Ensure that your `.api_key.json` and `.env` files are correctly configured with your API keys and database credentials. Both files are mandatory for the application to work properly.

## Memory System

This application includes a memory system that stores and retrieves user interactions. The memory system uses Supabase with the pgvector extension for semantic similarity search.

### Memory System Setup

The memory system is automatically initialized when you run the application for the first time. The memory system requires both:

1. Supabase credentials in `.api_key.json` for API-based operations
2. PostgreSQL connection details in `.env` for direct database operations

If you encounter any issues, you can manually set up the memory system by following the instructions in `supabase/SETUP_INSTRUCTIONS.md`.

### User Accounts

The application supports user accounts, which allows for personalized memory retrieval. When you start the application, you'll be prompted to enter a username. This username is used to associate memories with specific users.