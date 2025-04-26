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

3. Create a `.env` file in the project root with your database credentials:
   ```
   user=your_postgres_user
   password=your_postgres_password
   host=your_postgres_host
   port=your_postgres_port
   dbname=your_postgres_database
   ```
   This file is required for the memory system to work properly.

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Initialize the frontend submodule and install Node.js dependencies:
   ```bash
   git submodule update --init --recursive
   cd frontend
   npm install
   cd ..
   ```

6. Run the application (UI version):
   ```bash
   # Terminal 1: Start the API server
   python api_server.py

   # Terminal 2: Start the frontend
   cd frontend
   npm run dev
   ```

   Then open your browser to http://localhost:3000

   Alternatively, you can run the CLI version:
   ```bash
   python gpt_realtime/realtime_audio_gpt.py
   ```

The application will attempt to set up the database tables and functions on first run, but both the `.api_key.json` and `.env` files are required.

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

### Database Connection Requirements

The memory system requires both Supabase API access and direct PostgreSQL connection:

1. **Supabase API** (via `.api_key.json`): Used for vector similarity searches and some database operations
2. **Direct PostgreSQL** (via `.env`): Used for database initialization, memory storage, and advanced features

Both connection methods are required for the application to function properly. The `.env` file must contain:
```
user=your_postgres_user
password=your_postgres_password
host=your_postgres_host
port=your_postgres_port
dbname=your_postgres_database
```

## Features

- AI-powered conversations using OpenAI's GPT and Google's Gemini models
- Modern React-based UI with both chat and voice conversation modes
- User account system with personalized experiences
- Plutchik's wheel of emotions for real-time emotion visualization
- Emotion detection and realtime emotion analysis
- Natural language understanding
- Long-term memory with semantic search capabilities

## Technologies

- Python for backend processing and API server
- React.js for the frontend user interface
- Flask and Socket.IO for real-time communication between frontend and backend
- OpenAI real-time API for real-time audio conversations
- Google Gemini API and Google ADK for sentiment analysis and report generation
- Supabase with pgvector for memory storage and semantic search
- Node.js and npm for frontend development

## Running the Application

### UI Version (Recommended)

The application now includes a React-based UI that provides a more user-friendly interface with both chat and voice conversation modes.

#### Prerequisites

- Node.js and npm installed on your system
- All API keys and database configuration as described above

#### Setup and Run

1. Initialize the frontend submodule (first time only):
   ```bash
   git submodule update --init --recursive
   ```

2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. Start the API server:
   ```bash
   python api_server.py
   ```

4. In a separate terminal, start the frontend development server:
   ```bash
   cd frontend
   npm run dev
   ```

5. Open your browser and navigate to http://localhost:3000

The UI version includes:
- User login screen
- Text chat interface
- Real-time voice conversation with microphone support
- Plutchik's wheel of emotions visualization
- Memory integration with previous conversations

### CLI Audio Conversation Mode

The original command-line application uses OpenAI's real-time audio API for natural voice conversations:

```bash
python gpt_realtime/realtime_audio_gpt.py
```

This will:
- Automatically set up the database
- Prompt you for a username
- Start a voice conversation interface
- Enable real-time sentiment analysis

### Command Line Arguments

The CLI application supports several command line arguments:

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

The memory system attempts to initialize automatically when you run the application, but requires:

1. Supabase credentials in your `.api_key.json` file
2. PostgreSQL connection details in your `.env` file

Both files are mandatory for the memory system to function properly.

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