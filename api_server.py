"""
API Server for the Emotive Conversational Agent.

This server provides API endpoints for the React frontend to communicate with the backend.
"""

import os
import sys
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import Flask and related libraries
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Import backend components
from gpt_realtime.shared_emotion_state import SharedEmotionState, CORE_EMOTIONS
from memory.memory_integration import MemoryIntegration
from accounts.user_account_manager import UserAccountManager
from gpt_realtime.transcript_and_audio import TranscriptProcessor

# Import the RealtimeClient class
from gpt_realtime.realtime_audio_gpt import RealtimeClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load API keys
try:
    with open('.api_key.json', 'r') as f:
        api_keys = json.load(f)
        openai_api_key = api_keys.get('openai_api_key', '')
        gemini_api_key = api_keys.get('gemini_api_key', '')
except Exception as e:
    logger.error(f"Error loading API keys: {e}")
    openai_api_key = os.environ.get('OPENAI_API_KEY', '')
    gemini_api_key = os.environ.get('GEMINI_API_KEY', '')

# Initialize shared emotion state
shared_emotion_scores = SharedEmotionState()

# Initialize user account manager
user_manager = UserAccountManager()
user_manager.connect()

# Client instances dictionary (user_id -> client)
clients = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint."""
    data = request.json
    username = data.get('username', '')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Check if user exists, create if not
    user = user_manager.get_user_by_username(username)
    if not user:
        user = user_manager.register_user(username=username, display_name=username)

    return jsonify({
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name", username)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint for text-based conversations."""
    data = request.json
    message = data.get('message', '')
    user_id = data.get('user_id', '')
    username = data.get('username', '')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Get or create client for this user
    client = get_or_create_client(user_id, username)

    # Process the message asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Pass message as a parameter to the method
    response = loop.run_until_complete(client.send_chat_message_and_get_response(message=message))
    loop.close()

    return jsonify({
        "response": response,
        "emotions": get_current_emotions()
    })

@app.route('/api/audio', methods=['POST'])
def process_audio():
    """Process audio data and return a response."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    user_id = request.form.get('user_id', '')
    username = request.form.get('username', '')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Save the audio file temporarily
    temp_audio_path = f"temp_audio_{user_id}.webm"
    audio_file.save(temp_audio_path)

    # Get or create client for this user
    client = get_or_create_client(user_id, username)

    # Process the audio asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(client.process_audio_file(audio_file_path=temp_audio_path))
    loop.close()

    # Clean up the temporary file
    try:
        os.remove(temp_audio_path)
    except:
        pass

    return jsonify({
        "response": response,
        "emotions": get_current_emotions()
    })

@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get current emotion data."""
    return jsonify(get_current_emotions())

@app.route('/api/memories', methods=['GET'])
def get_memories():
    """Get memories for a user."""
    user_id = request.args.get('user_id', '')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Get or create client for this user
    client = get_or_create_client(user_id)

    # Get memories
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    memories = loop.run_until_complete(client.get_memories())
    loop.close()

    return jsonify({"memories": memories})

@app.route('/api/store-memory', methods=['POST'])
def store_memory():
    """Store a memory for a user."""
    data = request.json
    memory = data.get('memory', '')
    user_id = data.get('user_id', '')

    if not memory:
        return jsonify({"error": "Memory content is required"}), 400

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Get or create client for this user
    client = get_or_create_client(user_id)

    # Store memory
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(client.store_memory(memory_content=memory))
    loop.close()

    return jsonify({"success": success})

def get_or_create_client(user_id, username=None):
    """Get or create a client for the given user ID."""
    if user_id in clients:
        return clients[user_id]

    # Create a new client
    client = RealtimeClient(
        api_key=openai_api_key,
        google_api_key=gemini_api_key,
        enable_sentiment_analysis=True,
        shared_emotion_scores=shared_emotion_scores,
        conversation_mode="chat",  # Default to chat mode
        enable_memory=True,
        user_id=user_id,
        username=username
    )

    # Initialize memory integration with performance optimization
    if client.enable_memory and hasattr(client, 'memory_integration') and client.memory_integration:
        # Set optimize_performance flag to True for better performance
        client.memory_integration.optimize_performance = True
        logger.info("Memory performance optimization enabled")

    # Define methods directly on the client instance
    async def _send_chat_message_and_get_response(client_self, message):
        """Send a chat message and get the response."""
        return await send_chat_message_and_get_response(client=client_self, message=message)

    async def _process_audio_file(client_self, audio_file_path):
        """Process an audio file and return a response."""
        return await process_audio_file(client=client_self, audio_file_path=audio_file_path)

    async def _get_memories(client_self):
        """Get memories for the user."""
        return await get_user_memories(client=client_self)

    async def _store_memory(client_self, memory_content):
        """Store a memory for the user."""
        return await store_user_memory(client=client_self, memory_content=memory_content)

    # Attach methods to client instance
    client.send_chat_message_and_get_response = _send_chat_message_and_get_response.__get__(client)
    client.process_audio_file = _process_audio_file.__get__(client)
    client.get_memories = _get_memories.__get__(client)
    client.store_memory = _store_memory.__get__(client)

    # Store the client
    clients[user_id] = client

    return client

def get_current_emotions():
    """Get current emotion data from the shared state."""
    if shared_emotion_scores:
        user_scores, assistant_scores = shared_emotion_scores.get_emotion_scores()
        return {
            "user": user_scores,
            "assistant": assistant_scores
        }
    return {
        "user": {emotion: 0 for emotion in CORE_EMOTIONS},
        "assistant": {emotion: 0 for emotion in CORE_EMOTIONS}
    }

# Helper function to create async methods for the client
def async_method(func):
    """Decorator to make a function a method of the client instance."""
    async def wrapper(self, *args, **kwargs):
        # When attached to a client instance, 'self' will be the client
        return await func(client=self, **kwargs)
    return wrapper

# Async methods for the client
async def send_chat_message_and_get_response(client, message=None):
    """Send a chat message and get the response."""
    if message is None:
        raise ValueError("Message cannot be None")
    # Add to chat history
    client.chat_history.append({"role": "user", "content": message})

    # Process text for sentiment analysis if enabled
    if client.enable_sentiment_analysis:
        await client._process_text_for_sentiment(message, "user")

    # Add to transcript
    client.transcript_processor.add_user_query(message)

    # Process with memory if enabled
    if client.enable_memory and client.memory_integration:
        try:
            await client.memory_integration.process_user_message(message, client.conversation_mode)
        except Exception as e:
            logger.error(f"Error processing message with memory agent: {e}")

    # Prepare messages for the chat API
    messages = []

    # Create personalized instructions if username is available
    user_greeting = f"The user's name is {client.username}." if client.username else ""

    # Create system prompt
    system_prompt = f"""You are an empathetic AI assistant designed to help users process and understand their emotions.
{user_greeting}
You can remember information about the user across conversations.
When the user asks you to remember something, you will store it in your memory.
When the user asks about something you should know, you will search your memory for relevant information."""

    # Add system message
    system_message = {
        "role": "system",
        "content": system_prompt
    }
    messages.append(system_message)

    # Add chat history
    for msg in client.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Enhance with memories if enabled
    if client.enable_memory and client.memory_integration:
        try:
            enhanced_messages = await client.memory_integration.enhance_chat_messages(messages, message)
            messages = enhanced_messages
        except Exception as e:
            logger.error(f"Error enhancing messages with memories: {e}")

    # Call the chat completions API
    response = client.openai_client.chat.completions.create(
        model=client.chat_model,
        messages=messages,
        temperature=0.7
    )

    # Get the response text
    response_text = response.choices[0].message.content

    # Add to chat history
    client.chat_history.append({"role": "assistant", "content": response_text})

    # Process for sentiment analysis
    if client.enable_sentiment_analysis:
        await client._process_text_for_sentiment(response_text, "model")

    # Add to transcript
    client.transcript_processor.add_assistant_response(response_text)

    # Process with memory if enabled
    if client.enable_memory and client.memory_integration:
        try:
            await client.memory_integration.process_assistant_message(response_text, client.conversation_mode)
        except Exception as e:
            logger.error(f"Error processing assistant message with memory agent: {e}")

    return response_text

async def process_audio_file(client, audio_file_path=None):
    """Process an audio file and return a response."""
    if audio_file_path is None:
        raise ValueError("Audio file path cannot be None")
    # TODO: Implement audio processing
    # This is a placeholder for now
    return "Audio processing not yet implemented"

async def get_user_memories(client):
    """Get memories for the user."""
    if not client.enable_memory or not client.memory_integration:
        return []

    try:
        # Get memories from the memory manager
        memories = await client.memory_integration.memory_agent.memory_manager.list_memories(user_id=client.user_id)
        return memories
    except Exception as e:
        logger.error(f"Error getting memories: {e}")
        return []

async def store_user_memory(client, memory_content=None):
    """Store a memory for the user."""
    if memory_content is None:
        raise ValueError("Memory content cannot be None")

    if not client.enable_memory or not client.memory_integration:
        return False

    try:
        # Store memory
        await client.memory_integration.memory_agent.memory_manager.add_memory(
            content=memory_content,
            importance=8,  # Default importance
            category="user_input",
            user_id=client.user_id
        )
        return True
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
