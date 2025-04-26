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
import base64
import websockets
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import Flask and related libraries
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Import backend components
from gpt_realtime.shared_emotion_state import SharedEmotionState, CORE_EMOTIONS
from memory.memory_integration import MemoryIntegration
from accounts.user_account_manager import UserAccountManager
from gpt_realtime.transcript_and_audio import TranscriptProcessor

# Import the RealtimeClient class
from gpt_realtime.realtime_audio_gpt import RealtimeClient

# Create a subclass of RealtimeClient that emits events to the WebSocket
class WebSocketRealtimeClient(RealtimeClient):
    """RealtimeClient that emits events to the WebSocket."""

    def __init__(self, *args, **kwargs):
        """Initialize the WebSocketRealtimeClient."""
        super().__init__(*args, **kwargs)
        self.is_disconnected = False
        self.session_id = None
        self.audio_ready = False  # Flag to track if audio components are initialized

        # Add required properties for WebSocket connection
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-realtime-preview-2024-12-17"  # Realtime model
        self.voice = "shimmer"  # Default voice

        # Server-side VAD configuration
        self.VAD_turn_detection = True
        self.VAD_config = {
            "type": "server_vad",
            "threshold": 0.3,  # Lower threshold to make it more sensitive
            "prefix_padding_ms": 500,  # Padding before speech
            "silence_duration_ms": 1000,  # Silence duration to detect end of speech
            "create_response": True,  # Explicitly enable auto-response creation
            "interrupt_response": True  # Enable interrupting the model's response
        }

        # Session configuration
        # Create personalized instructions if username is available
        user_greeting = f"The user's name is {self.username}." if self.username else ""

        # Create a more detailed session configuration with explicit voice setting
        self.session_config = {
            "modalities": ["audio", "text"],
            "voice": "shimmer",  # Explicitly set to shimmer
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "instructions": f"You are a helpful AI assistant with access to memories from previous conversations. {user_greeting} These memories will be provided to you as system messages. Use these memories to provide personalized and contextually relevant responses. When appropriate, reference these past interactions naturally. Respond in a clear and engaging way.",
            "turn_detection": self.VAD_config,  # Use the VAD configuration defined above
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "temperature": 0.7
        }

        # Log the VAD configuration for debugging
        logger.info(f"VAD configuration: {self.VAD_config}")

        # Audio state
        self.audio_buffer = b''  # Buffer for streaming audio responses
        self.is_responding = False
        self.response_complete_event = asyncio.Event()
        self.response_complete_event.set()  # Initially set to True (not responding)

        # Silence detection parameters
        self.silence_threshold = 100  # Amplitude threshold for silence detection
        self.silence_duration_threshold = 2.0  # Seconds of silence before considering it a silence period
        self.last_audio_activity_time = time.time()  # Track the last time audio activity was detected
        self.silence_check_interval = 1.0  # How often to check for silence (seconds)
        self.last_silence_check_time = time.time()  # Last time we checked for silence

        # Sentiment analysis parameters
        self.sentiment_model_name = "models/gemini-2.0-flash"
        self.sentiment_file_path = None  # Path to save sentiment results
        self.sentiment_file_saved = False  # Flag to track if we've saved the sentiment file

    def _update_sentiment_file(self, emotion_scores, source):
        """Update the sentiment scores in the shared emotion state.

        Args:
            emotion_scores (dict): The emotion scores to update
            source (str): Either 'user' or 'assistant'
        """
        try:
            # Update the shared emotion state if available
            if self.enable_sentiment_analysis and self.sentiment_manager and self.sentiment_manager.shared_emotion_scores:
                # Use the SharedEmotionState's update_emotion_scores method
                success = self.sentiment_manager.shared_emotion_scores.update_emotion_scores(source, emotion_scores)
                if success:
                    print(f"[SENTIMENT ANALYSIS] Updated {source} emotion scores in shared state")
                else:
                    print(f"[SENTIMENT ANALYSIS] Failed to update {source} emotion scores in shared state")
            else:
                print(f"[SENTIMENT ANALYSIS] Cannot update {source} emotion scores: sentiment analysis not enabled or manager not available")
        except Exception as e:
            logger.error(f"Error updating sentiment file: {e}")

    async def handle_event(self, event):
        """Override handle_event to emit events to the WebSocket."""
        try:
            # First, call the parent's handle_event method
            await super().handle_event(event)

            # Then emit the event to the WebSocket if not disconnected
            if not self.is_disconnected:
                emit_realtime_events(self, event)
        except Exception as e:
            logger.error(f"Error in WebSocketRealtimeClient.handle_event: {e}")
            # Try to continue processing despite errors

    def reset_audio_state(self):
        """Reset audio state when interrupted."""
        if hasattr(self, 'is_responding') and self.is_responding:
            logger.info("Resetting audio state")
            self.is_responding = False
            if hasattr(self, 'response_complete_event'):
                self.response_complete_event.set()

            # Clear any buffered audio
            if hasattr(self, 'audio_buffer'):
                self.audio_buffer = b''

            logger.info("Audio state reset complete")

    async def send_event(self, event):
        """Send an event to the WebSocket server."""
        # Check if WebSocket exists
        if self.ws:
            # Check if it has a closed attribute and if so, check if it's closed
            is_closed = False
            if hasattr(self.ws, 'closed'):
                is_closed = self.ws.closed

            if not is_closed:
                try:
                    await self.ws.send(json.dumps(event))
                    logger.debug(f"Sent event: {event.get('type')}")
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending event: {e}")
                    # Try to reconnect
                    logger.info("Attempting to reconnect...")
                    # Clear the existing connection
                    self.ws = None
                    # Try to initialize audio components again (which will reconnect)
                    reconnected = await self.initialize_audio()

                    # Check if reconnection was successful
                    if reconnected and self.ws:
                        # Check if it has a closed attribute and if so, check if it's closed
                        is_reconnect_closed = False
                        if hasattr(self.ws, 'closed'):
                            is_reconnect_closed = self.ws.closed

                        if not is_reconnect_closed:
                            logger.info("Successfully reconnected, retrying event send")
                            # Try sending the event again
                            await self.ws.send(json.dumps(event))
                        else:
                            logger.error("Reconnected but WebSocket is closed")
                            raise RuntimeError("Reconnected but WebSocket is closed")
                    else:
                        logger.error("Failed to reconnect")
                        raise RuntimeError("Failed to reconnect to WebSocket server")
                except asyncio.CancelledError:
                    logger.error("Event sending was cancelled (event loop closed)")
                    raise  # Re-raise to allow proper handling
                except Exception as e:
                    logger.error(f"Error sending event: {e}")
                    if "Event loop is closed" in str(e):
                        logger.error("Event loop is closed, cannot send event")
                        # Don't try to reconnect here, as we need a new event loop
                        raise RuntimeError("Event loop is closed, cannot send event")
                    # Re-raise to allow proper error handling by caller
                    raise
            else:
                logger.error("Cannot send event: WebSocket is closed")
                raise RuntimeError("WebSocket connection is closed")
        else:
            logger.error("Cannot send event: WebSocket is not connected")
            raise RuntimeError("WebSocket connection is not established")

    async def _process_text_for_sentiment(self, text, source):
        """Process text for sentiment analysis.

        Args:
            text (str): The text to analyze
            source (str): Either 'user' or 'model'
        """
        if not self.enable_sentiment_analysis or not self.sentiment_manager:
            return

        try:
            # Send text to the sentiment analysis manager
            self.sentiment_manager.send_text(text, source)
            logger.info(f"Sent {source} text for sentiment analysis: {text[:30]}...")
        except Exception as e:
            logger.error(f"Error sending text for sentiment analysis: {e}")

    def _set_assistant_sentiment_scores(self):
        """Analyze the assistant's response text for sentiment and update the scores.

        This method extracts the assistant's response text and sends it for sentiment analysis.
        It also directly updates the shared emotion scores dictionary for immediate feedback.
        """
        if not self.enable_sentiment_analysis or not self.sentiment_manager:
            return

        try:
            # Get the assistant's response text
            assistant_text = ""
            if hasattr(self, 'current_response_text') and self.current_response_text:
                assistant_text = self.current_response_text
            elif hasattr(self, 'transcript_processor') and self.transcript_processor:
                # Try to get the last assistant response from the transcript
                assistant_text = self.transcript_processor.get_last_assistant_response()

            # If we have text, analyze it directly and update scores immediately
            if assistant_text:
                # For immediate feedback, directly analyze a small portion of the text
                # This gives us quick updates while the full analysis happens in the background
                print(f"\n[SENTIMENT ANALYSIS] Analyzing assistant text: {assistant_text[:50]}...")

                # Send the full text for proper sentiment analysis in the background
                self.sentiment_manager.send_text(assistant_text, "model")

                # For immediate feedback, create estimated scores based on text content
                # This is a simple heuristic approach that will be replaced by the proper analysis
                estimated_scores = self._quick_estimate_sentiment(assistant_text)

                # Update the shared emotion scores with our estimated scores
                if self.sentiment_manager.shared_emotion_scores:
                    self.sentiment_manager.shared_emotion_scores.update_emotion_scores('assistant', estimated_scores)
                    print(f"\n[SENTIMENT ANALYSIS] Updated assistant emotion scores with estimates: {estimated_scores}")

                # Also update the file for the radar chart
                self._update_sentiment_file(estimated_scores, 'assistant')
                return

            # If we don't have text, use a fallback approach with estimated scores
            # based on the context of the conversation
            assistant_scores = {
                'Joy': 1,  # Assistants are generally positive
                'Trust': 2,  # Assistants aim to build trust
                'Fear': 0,   # Assistants rarely express fear
                'Surprise': 0,
                'Sadness': 0,
                'Disgust': 0,
                'Anger': 0,
                'Anticipation': 1  # Assistants often express anticipation
            }

            # Update the shared dictionary with our fallback scores
            if self.sentiment_manager.shared_emotion_scores:
                self.sentiment_manager.shared_emotion_scores.update_emotion_scores('assistant', assistant_scores)
                print(f"\n[SENTIMENT ANALYSIS] Set fallback assistant emotion scores: {assistant_scores}")

            # Also update the file for the radar chart
            self._update_sentiment_file(assistant_scores, 'assistant')

        except Exception as e:
            logger.error(f"Error setting assistant sentiment scores: {e}")

    def _quick_estimate_sentiment(self, text):
        """Quickly estimate sentiment scores based on text content.

        This is a simple heuristic approach for immediate feedback while the proper
        sentiment analysis happens in the background.

        Args:
            text (str): The text to analyze

        Returns:
            dict: Estimated emotion scores
        """
        # Default scores - assistants are generally positive and trustworthy
        scores = {
            'Joy': 1,
            'Trust': 2,
            'Fear': 0,
            'Surprise': 0,
            'Sadness': 0,
            'Disgust': 0,
            'Anger': 0,
            'Anticipation': 1
        }

        # Simple keyword-based heuristics
        text_lower = text.lower()

        # Joy indicators
        if any(word in text_lower for word in ['happy', 'glad', 'joy', 'wonderful', 'excellent', 'great']):
            scores['Joy'] = min(scores['Joy'] + 0.5, 3.0)

        # Trust indicators
        if any(word in text_lower for word in ['trust', 'reliable', 'confident', 'certain']):
            scores['Trust'] = min(scores['Trust'] + 0.5, 3.0)

        # Fear indicators
        if any(word in text_lower for word in ['afraid', 'fear', 'scary', 'worried', 'concerned']):
            scores['Fear'] = min(scores['Fear'] + 0.5, 3.0)

        # Surprise indicators
        if any(word in text_lower for word in ['surprise', 'wow', 'amazing', 'unexpected', 'astonishing']):
            scores['Surprise'] = min(scores['Surprise'] + 0.5, 3.0)

        # Sadness indicators
        if any(word in text_lower for word in ['sad', 'sorry', 'unfortunate', 'regret']):
            scores['Sadness'] = min(scores['Sadness'] + 0.5, 3.0)

        # Disgust indicators
        if any(word in text_lower for word in ['disgust', 'gross', 'unpleasant']):
            scores['Disgust'] = min(scores['Disgust'] + 0.5, 3.0)

        # Anger indicators
        if any(word in text_lower for word in ['angry', 'frustrating', 'annoying']):
            scores['Anger'] = min(scores['Anger'] + 0.5, 3.0)

        # Anticipation indicators
        if any(word in text_lower for word in ['looking forward', 'anticipate', 'expect', 'hope']):
            scores['Anticipation'] = min(scores['Anticipation'] + 0.5, 3.0)

        return scores

    def _check_for_silence(self, chunk):
        """Check if the audio chunk is silent and update state accordingly."""
        try:
            # Convert the chunk to a numpy array
            audio_array = np.frombuffer(chunk, dtype=np.int16)

            # Calculate the maximum amplitude
            max_amp = np.max(np.abs(audio_array))

            # Get the current time
            current_time = time.time()

            # Check if this is a silent chunk
            if max_amp < getattr(self, 'silence_threshold', 100):  # Default threshold of 100
                # If we haven't set the last_audio_activity_time, set it now
                if not hasattr(self, 'last_audio_activity_time'):
                    self.last_audio_activity_time = current_time

                # Check if we've been silent for long enough
                silence_duration = current_time - self.last_audio_activity_time
                if silence_duration > getattr(self, 'silence_duration_threshold', 2.0):  # Default 2 seconds
                    # We've been silent for long enough, check if we should log it
                    if not hasattr(self, 'last_silence_check_time') or current_time - self.last_silence_check_time > getattr(self, 'silence_check_interval', 1.0):
                        logger.debug(f"Silence detected for {silence_duration:.1f} seconds")
                        self.last_silence_check_time = current_time

                    # Update the shared emotion state if available
                    if self.enable_sentiment_analysis and self.sentiment_manager and self.sentiment_manager.shared_emotion_scores:
                        # Only update if the state has changed
                        current_silence_state = self.sentiment_manager.shared_emotion_scores.is_silence_detected()
                        if not current_silence_state:
                            self.sentiment_manager.shared_emotion_scores.set_silence_detected(True)
                            # Also set VAD speech detection flag to false when silence is detected
                            self.sentiment_manager.shared_emotion_scores.set_vad_speech_detected(False)
                            logger.info(f"Silence detected after {silence_duration:.1f} seconds of inactivity")
            else:
                # This chunk has audio activity, update the last activity time
                self.last_audio_activity_time = current_time

                # Update the shared emotion state if available
                if self.enable_sentiment_analysis and self.sentiment_manager and self.sentiment_manager.shared_emotion_scores:
                    # Set silence detected to false when audio activity is detected
                    self.sentiment_manager.shared_emotion_scores.set_silence_detected(False)
                    # Set VAD speech detection flag to true when audio activity is detected
                    self.sentiment_manager.shared_emotion_scores.set_vad_speech_detected(True)
                    # Update the last audio activity time
                    self.sentiment_manager.shared_emotion_scores.update_audio_activity_time()
                    logger.debug("Audio activity detected, updating shared emotion state")
        except Exception as e:
            logger.error(f"Error in _check_for_silence: {e}")
            # Don't propagate the exception, just log it

    async def initialize_audio(self):
        """Initialize audio components."""
        try:
            logger.info(f"Initializing audio components for user {self.username} (ID: {self.user_id})")

            # Connect to the OpenAI WebSocket if not already connected
            # Check if ws exists and is connected in a way that works with different WebSocket libraries
            # Use a safer approach to check if the connection is closed
            ws_connected = hasattr(self, 'ws') and self.ws is not None
            # Only check the closed attribute if it exists
            if ws_connected and hasattr(self.ws, 'closed'):
                ws_connected = not self.ws.closed

            if not ws_connected:
                logger.info("WebSocket not connected, attempting to connect...")
                try:
                    # Create the URL with the model parameter in the query string
                    full_url = f"{self.url}?model={self.model}"

                    # Prepare headers dictionary
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "OpenAI-Beta": "realtime=v1"
                    }

                    # Connect to the WebSocket with proper parameters
                    self.ws = await websockets.connect(
                        full_url,
                        additional_headers=headers,
                        ping_interval=20,
                        ping_timeout=60
                    )

                    logger.info("Successfully connected to OpenAI Realtime API")

                    # Configure session - this is critical for the connection to work properly
                    try:
                        # Use the session configuration defined in __init__
                        # Make sure the voice is explicitly set to shimmer
                        self.session_config["voice"] = "shimmer"

                        # Send the session configuration
                        await self.send_event({
                            "type": "session.update",
                            "session": self.session_config
                        })
                        logger.info(f"Session configuration sent with voice={self.session_config['voice']}")

                        # Wait for acknowledgment
                        try:
                            session_ack = await asyncio.wait_for(self.ws.recv(), timeout=5)
                            session_ack_json = json.loads(session_ack)

                            # Check for session.created response (initial connection)
                            if session_ack_json.get("type") == "session.created":
                                session_data = session_ack_json.get("session", {})
                                voice = session_data.get("voice")
                                session_id = session_data.get("id")

                                # Store the session ID for future reference
                                if session_id:
                                    self.session_id = session_id
                                    logger.info(f"Session ID: {session_id}")

                                if voice == "shimmer":
                                    logger.info(f"Session successfully initialized with voice: {voice}")
                                else:
                                    logger.warning(f"Session created with unexpected voice: {voice}, expected 'shimmer'")

                                    # Try to update the session again with explicit voice setting
                                    logger.info("Attempting to update session with explicit voice setting...")
                                    await self.send_event({
                                        "type": "session.update",
                                        "session": {"voice": "shimmer"}
                                    })

                                    # Wait for the update response
                                    try:
                                        update_ack = await asyncio.wait_for(self.ws.recv(), timeout=3)
                                        update_ack_json = json.loads(update_ack)
                                        if update_ack_json.get("type") == "session.updated":
                                            logger.info("Voice successfully updated to 'shimmer'")
                                        else:
                                            logger.warning(f"Unexpected response to voice update: {update_ack_json}")
                                    except Exception as e:
                                        logger.warning(f"Error waiting for voice update response: {e}")
                            # Check for session.updated response (after update)
                            elif session_ack_json.get("type") == "session.updated":
                                logger.info("Session successfully updated")
                            else:
                                logger.warning(f"Unexpected session response: {session_ack_json}")

                                # Try to update the session again with just the voice parameter
                                logger.info("Attempting to update session with minimal configuration...")
                                await self.send_event({
                                    "type": "session.update",
                                    "session": {"voice": "shimmer"}
                                })
                        except asyncio.TimeoutError:
                            logger.warning("No session acknowledgment received within timeout")
                            # Try a simpler update with just the voice parameter
                            logger.info("Attempting voice-only session update after timeout...")
                            await self.send_event({
                                "type": "session.update",
                                "session": {"voice": "shimmer"}
                            })
                        except Exception as e:
                            logger.error(f"Error receiving session acknowledgment: {e}")
                            # Continue anyway as some implementations might not send an acknowledgment

                    except Exception as e:
                        logger.error(f"Failed to send session configuration: {e}")
                        if self.ws:
                            await self.ws.close()
                        self.ws = None
                        return False

                    # After connect, check if we have a valid ws object
                    ws_connected = hasattr(self, 'ws') and self.ws is not None
                    # Only check the closed attribute if it exists
                    if ws_connected and hasattr(self.ws, 'closed'):
                        ws_connected = not self.ws.closed

                    if not ws_connected:
                        logger.error("Failed to establish WebSocket connection")
                        return False
                    logger.info("WebSocket connection established successfully")
                except Exception as connect_error:
                    logger.error(f"Error connecting to WebSocket: {connect_error}")
                    return False
            else:
                logger.info("WebSocket already connected")

            # Reset audio state
            self.reset_audio_state()

            # Check if the WebSocket is still connected before sending a test event
            ws_connected = hasattr(self, 'ws') and self.ws is not None
            # Only check the closed attribute if it exists
            if ws_connected and hasattr(self.ws, 'closed'):
                ws_connected = not self.ws.closed

            if ws_connected:
                try:
                    # Send a small test event to ensure the connection is working
                    logger.info("Sending test event to WebSocket")
                    await self.send_event({"type": "ping"})
                    logger.info("Test event sent successfully")
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Connection closed while sending test event: {e}")
                    # Try to reconnect
                    logger.info("Connection closed, attempting to reconnect...")
                    self.ws = None  # Clear the existing connection
                    return await self.initialize_audio()  # Recursive call to retry initialization
                except Exception as event_error:
                    logger.error(f"Error sending test event: {event_error}")
                    # Don't fail the whole initialization if just the test event fails
                    # The connection might still be usable
            else:
                logger.warning("WebSocket not connected, skipping test event")

            # Mark audio as ready
            self.audio_ready = True
            logger.info(f"Audio components initialized successfully for user {self.username}")
            return True
        except Exception as e:
            logger.error(f"Error initializing audio components: {e}")
            # Set audio_ready to false to ensure we don't think it's ready
            self.audio_ready = False
            return False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,  # Increase ping timeout to 60 seconds
    ping_interval=25,  # Send pings more frequently
    max_http_buffer_size=50 * 1024 * 1024,  # Increase buffer size for large audio chunks
    engineio_logger=True  # Enable Engine.IO logging for debugging
)

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

# Set an environment variable to detect duplicate initialization
import os
print("API Server starting...")
if os.environ.get('SUPABASE_INITIALIZED') == 'true':
    print("Detected duplicate initialization, skipping...")
else:
    os.environ['SUPABASE_INITIALIZED'] = 'true'
    print("First initialization, proceeding...")

# Initialize shared emotion state
shared_emotion_scores = SharedEmotionState()

# Initialize user account manager
user_manager = UserAccountManager()

# Client instances dictionary (user_id -> client)
clients = {}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/api/pre-initialize', methods=['POST'])
def pre_initialize():
    """Pre-initialize endpoint to warm up the system after login."""
    data = request.json
    user_id = data.get('user_id', '')
    username = data.get('username', '')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Get or create client for this user
    client = get_or_create_client(user_id, username)

    # Only start initialization if it's not already in progress
    if not client.is_initializing and not client.is_initialized:
        # Set initializing flag
        client.is_initializing = True
        # Start pre-initialization in a background thread
        threading.Thread(target=pre_initialize_client_background, args=(user_id, username), daemon=True).start()
        logger.info(f"Started pre-initialization for user {username} (ID: {user_id})")
    else:
        logger.info(f"Pre-initialization already in progress or completed for user {username} (ID: {user_id})")

    # Return current initialization status
    return jsonify({
        "status": "initialization_started",
        "is_initializing": client.is_initializing,
        "is_initialized": client.is_initialized
    })

# Track initialization status check timestamps to prevent rapid duplicate checks
initialization_status_timestamps = {}

@app.route('/api/initialization-status', methods=['GET'])
def initialization_status():
    """Check the initialization status for a user."""
    user_id = request.args.get('user_id', '')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Check if we've recently processed a status check for this user
    current_time = time.time()
    if user_id in initialization_status_timestamps:
        last_check_time = initialization_status_timestamps[user_id]
        time_since_last_check = current_time - last_check_time

        # If it's been less than 3 seconds since the last check, use cached result
        if time_since_last_check < 3 and hasattr(initialization_status, 'last_result') and initialization_status.last_result.get('user_id') == user_id:
            logger.debug(f"Using cached initialization status for {user_id} (last check was {time_since_last_check:.2f} seconds ago)")
            return jsonify(initialization_status.last_result)

    # Update the check timestamp for this user
    initialization_status_timestamps[user_id] = current_time

    # Check if client exists
    if user_id not in clients:
        result = {
            "user_id": user_id,
            "is_initializing": False,
            "is_initialized": False,
            "audio_ready": False,
            "status": "not_started"
        }
        # Cache the result
        if not hasattr(initialization_status, 'last_result'):
            initialization_status.last_result = {}
        initialization_status.last_result = result
        return jsonify(result)

    client = clients[user_id]

    # Prepare the result
    result = {
        "user_id": user_id,
        "is_initializing": client.is_initializing,
        "is_initialized": client.is_initialized,
        "audio_ready": getattr(client, 'audio_ready', False),
        "status": "initializing" if client.is_initializing else "initialized" if client.is_initialized else "not_started"
    }

    # Cache the result
    if not hasattr(initialization_status, 'last_result'):
        initialization_status.last_result = {}
    initialization_status.last_result = result

    # Return initialization status
    return jsonify(result)

@app.route('/api/login', methods=['POST'])
def login():
    """Login endpoint."""
    data = request.json
    username = data.get('username', '')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Ensure we have a valid connection to Supabase
    if not user_manager.connect():
        return jsonify({"error": "Failed to connect to database"}), 500

    # Check if user exists, create if not
    user = user_manager.get_user_by_username(username)
    if not user:
        # Try to register a new user
        user = user_manager.register_user(username=username, display_name=username)
        if not user:
            return jsonify({"error": "Failed to register user"}), 500

    if not user:
        return jsonify({"error": "Failed to login or register user"}), 500

    # Start pre-initialization in the background
    # This will create the client but won't block the login response
    user_id = user["id"]

    # Reset client initialization state if it exists
    if user_id in clients:
        clients[user_id].is_initializing = False
        clients[user_id].is_initialized = False
        clients[user_id].first_message_processed = False
        logger.info(f"Reset initialization state for existing client: {username} (ID: {user_id})")

    # Start pre-initialization
    threading.Thread(target=pre_initialize_client_background, args=(user_id, username), daemon=True).start()

    return jsonify({
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user.get("display_name", username)
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout endpoint."""
    data = request.json
    user_id = data.get('user_id', '')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Ensure we have a valid connection to Supabase
    if not user_manager.connect():
        logger.warning("Failed to connect to database during logout, continuing with local logout")

    # Log out the user in the user manager
    user_manager.logout()

    # Reset client state but don't remove it from the clients dictionary
    # This allows us to reuse the client when the user logs back in
    if user_id in clients:
        client = clients[user_id]

        # Save sentiment results if enabled
        if hasattr(client, 'enable_sentiment_analysis') and client.enable_sentiment_analysis:
            if hasattr(client, 'sentiment_manager') and client.sentiment_manager:
                try:
                    logger.info("Saving sentiment analysis results before logout...")

                    # Save sentiment results if we haven't already
                    if hasattr(client, 'sentiment_file_saved') and not client.sentiment_file_saved and hasattr(client, 'sentiment_file_path') and client.sentiment_file_path:
                        # Stop the process and save to the user-specific folder using our consistent file path
                        client.sentiment_manager.stop(custom_file_path=client.sentiment_file_path)
                        logger.info(f"Saved sentiment analysis results to: {client.sentiment_file_path}")
                        client.sentiment_file_saved = True
                    else:
                        # Either we've already saved or we don't have a file path
                        # Fall back to the default path if needed
                        client.sentiment_manager.stop()
                        if hasattr(client, 'sentiment_file_saved'):
                            client.sentiment_file_saved = True
                except Exception as e:
                    logger.error(f"Error saving sentiment results: {e}")

        # Reset initialization flags
        client.is_initializing = False
        client.is_initialized = False
        client.first_message_processed = False
        logger.info(f"Reset client state for user ID: {user_id}")

        # If the client has a memory integration, reset it
        if hasattr(client, 'memory_integration') and client.memory_integration:
            # Clear any cached data
            if hasattr(client.memory_integration, 'last_retrieval_message'):
                client.memory_integration.last_retrieval_message = None
            if hasattr(client.memory_integration, 'last_retrieval_result'):
                client.memory_integration.last_retrieval_result = []

    return jsonify({"success": True})

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

    # Check if this is the first message and the client is still initializing
    is_first_message = not client.first_message_processed
    initialization_status = "initializing" if client.is_initializing else "initialized" if client.is_initialized else "not_initialized"

    # Log the status for debugging
    if is_first_message:
        logger.info(f"Processing first message for user {username} (ID: {user_id}). Initialization status: {initialization_status}")

    # Process the message asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Pass message as a parameter to the method
    response = loop.run_until_complete(client.send_chat_message_and_get_response(message=message))
    loop.close()

    return jsonify({
        "response": response,
        "emotions": get_current_emotions(),
        "is_first_message": is_first_message,
        "initialization_status": initialization_status
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

    # For now, we don't have a direct audio URL to return since the audio is played directly
    # by the RealtimeClient. In a future enhancement, we could save the audio response to a file
    # and return a URL to it.

    return jsonify({
        "response": response,
        "emotions": get_current_emotions(),
        # Note: The audio is currently played directly by the backend.
        # In the future, we could return an audio URL here for the frontend to play.
        "audio_played": True
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
    client = WebSocketRealtimeClient(
        api_key=openai_api_key,
        google_api_key=gemini_api_key,
        enable_sentiment_analysis=True,
        shared_emotion_scores=shared_emotion_scores,
        conversation_mode="audio",  # Use audio mode for real-time streaming
        enable_memory=True,
        user_id=user_id,
        username=username
    )

    # Ensure sentiment analysis is properly initialized
    if client.enable_sentiment_analysis and not hasattr(client, 'sentiment_manager') or not client.sentiment_manager:
        try:
            # Import the SentimentAnalysisManager class
            from gpt_realtime.sentiment_analyzer import SentimentAnalysisManager

            # Create the sentiment analysis manager
            client.sentiment_manager = SentimentAnalysisManager(
                google_api_key=gemini_api_key,
                sentiment_model_name=client.sentiment_model_name
            )
            logger.info(f"Sentiment analysis manager initialized for model {client.sentiment_model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analysis manager: {e}")
            client.enable_sentiment_analysis = False

    # Add initialization flags
    client.is_initializing = False
    client.is_initialized = False
    client.first_message_processed = False

    # Initialize memory integration with performance optimization and strict intent detection
    if client.enable_memory and hasattr(client, 'memory_integration') and client.memory_integration:
        # Set optimize_performance flag to True for better performance
        client.memory_integration.optimize_performance = True
        # Set strict_intent_detection flag to True to only retrieve memories when explicit intent is detected
        client.memory_integration.strict_intent_detection = True
        logger.info("Memory performance optimization and strict intent detection enabled")

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

def pre_initialize_client_background(user_id, username):
    """Pre-initialize a client for the given user ID in the background.
    This function is meant to be run in a separate thread to avoid blocking the login response.
    It performs a thorough initialization of all components to reduce lag on the first message.
    """
    logger.info(f"Pre-initializing client for user {username} (ID: {user_id})")

    try:
        # Get or create the client
        client = get_or_create_client(user_id, username)

        # Set initialization flag
        client.is_initializing = True

        # Pre-warm OpenAI API connection with a more substantial request
        try:
            logger.info("Pre-warming OpenAI API connection...")
            # Make a more substantial API call to properly warm up the connection
            client.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=["Hello", "How are you?", "I'm doing well", "What can you help me with today?"],
                dimensions=1536  # Use full dimensions
            )
            logger.info("OpenAI API connection established successfully.")

            # Also pre-warm the chat model
            logger.info("Pre-warming chat model...")
            client.openai_client.chat.completions.create(
                model=client.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, this is a pre-initialization test."}
                ],
                max_tokens=20  # Keep it small for efficiency
            )
            logger.info("Chat model pre-warmed successfully.")
        except Exception as e:
            logger.warning(f"Failed to pre-warm OpenAI API connection: {e}")

        # Pre-initialize sentiment analysis if enabled
        if client.enable_sentiment_analysis and hasattr(client, 'sentiment_manager') and client.sentiment_manager:
            try:
                logger.info("Pre-initializing sentiment analysis system...")

                # Set up the sentiment file path if not already set
                if not client.sentiment_file_path:
                    # Create a user-specific directory for transcripts and sentiment results
                    if hasattr(client, 'transcript_processor'):
                        if not hasattr(client.transcript_processor, 'user_transcripts_dir'):
                            # Create a user-specific directory for transcripts
                            user_dir = f"transcripts/{client.username}" if client.username else f"transcripts/user_{client.user_id}"
                            os.makedirs(user_dir, exist_ok=True)
                            client.transcript_processor.user_transcripts_dir = user_dir
                            logger.info(f"Created user transcripts directory: {user_dir}")

                        # Use the same timestamp as the conversation for consistent file naming
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        client.sentiment_file_path = os.path.join(client.transcript_processor.user_transcripts_dir, f'sentiment_results_{timestamp}.json')
                        logger.info(f"Sentiment results will be saved to: {client.sentiment_file_path}")

                # Start the sentiment analysis process with the file path
                if not client.sentiment_manager.is_running():
                    if client.sentiment_manager.start(client.sentiment_file_path):
                        logger.info("Sentiment analysis process started successfully.")
                    else:
                        logger.warning("Failed to start sentiment analysis process.")

                # Send a dummy text to warm up the sentiment analysis system
                client.sentiment_manager.send_text("Hello, this is a test message.", "user")
                logger.info("Sentiment analysis system initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to pre-initialize sentiment analysis: {e}")
                logger.warning(f"Note: Sentiment analysis pre-initialization failed, but this won't affect functionality.")

        # Pre-initialize memory system if enabled
        if client.enable_memory and client.memory_integration:
            try:
                logger.info("Pre-initializing memory system...")
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Perform a more realistic memory retrieval to warm up the system
                    # Use a variety of messages to warm up different parts of the system
                    dummy_messages = [
                        "Hello, how are you today?",
                        "I live in New York",  # Location-related to trigger that path
                        "Let me tell you about myself",  # Storage-related
                        "What did I tell you about my job?"  # Retrieval-related
                    ]

                    # Process each message to warm up different paths
                    for msg in dummy_messages:
                        logger.info(f"Pre-warming memory system with: '{msg}'")
                        # Force retrieval to ensure the database connection is established
                        loop.run_until_complete(client.memory_integration.get_context_with_memories(msg, force_retrieval=True))

                    # Also pre-warm the memory storage path
                    loop.run_until_complete(client.memory_integration.store_memory_directly(
                        content="This is a test memory for pre-initialization",
                        importance=5,
                        category="system"
                    ))

                    logger.info("Memory system initialized successfully.")
                except Exception as e:
                    logger.warning(f"Error during memory system initialization: {e}")
                finally:
                    # Always close the loop
                    try:
                        loop.close()
                    except Exception as close_error:
                        logger.warning(f"Error closing memory initialization event loop: {close_error}")
            except Exception as e:
                logger.warning(f"Failed to pre-initialize memory system: {e}")
                logger.warning(f"Error details: {str(e)}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")

        # Pre-initialize audio components
        try:
            logger.info("Pre-initializing audio components...")
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Initialize audio components
                audio_success = loop.run_until_complete(client.initialize_audio())

                if audio_success:
                    logger.info("Audio components initialized successfully.")
                else:
                    logger.warning("Failed to initialize audio components.")
            except Exception as e:
                logger.warning(f"Error during audio initialization: {e}")
                audio_success = False
            finally:
                # Always close the loop
                try:
                    loop.close()
                except Exception as close_error:
                    logger.warning(f"Error closing audio initialization event loop: {close_error}")

            # Set audio_ready flag based on success
            client.audio_ready = audio_success
        except Exception as e:
            logger.warning(f"Failed to pre-initialize audio components: {e}")
            logger.warning(f"Error details: {str(e)}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
            # Make sure audio_ready is set to False on error
            client.audio_ready = False

        # Mark initialization as complete
        client.is_initializing = False
        client.is_initialized = True
        logger.info(f"Client pre-initialization complete for user {username}")
    except Exception as e:
        logger.error(f"Error during client pre-initialization: {e}")
        # Make sure to reset the initialization flag even on error
        if 'client' in locals():
            client.is_initializing = False
            # Also make sure audio_ready is set to False on error
            client.audio_ready = False

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
    async def wrapper(self, **kwargs):
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
        # Start memory processing in a background thread to avoid blocking the response
        threading.Thread(
            target=process_memory_background,
            args=(client, message, client.conversation_mode),
            daemon=True
        ).start()
        logger.info("Started memory processing in background thread")

    # Prepare messages for the chat API
    messages = []

    # Create personalized instructions if username is available
    user_greeting = f"The user's name is {client.username}." if client.username else ""

    # Create system prompt
    system_prompt = f"""You are an empathetic AI assistant designed to help users process and understand their emotions.
{user_greeting}
You can remember information about the user across conversations.
When the user asks you to remember something, express your intent to remember it (e.g., "I'll remember that") - a background orchestrator will handle the actual storage.
When you want to recall information, use phrases like "let me recall" or "let me see what I know about you" - a background orchestrator will retrieve relevant memories.
DO NOT use function calls or tools to store or retrieve memories - just express your intent naturally in your response."""

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
            # Check if this is the first message and the client is still initializing
            if not client.first_message_processed and (client.is_initializing or not client.is_initialized):
                logger.info("First message detected while system is still initializing - using fast path")
                # For the first message during initialization, use a fast path with minimal memory operations
                # This ensures the first response is quick, even if it doesn't have full memory context
                # Subsequent messages will have the full memory context
            else:
                # Normal path for subsequent messages or if initialization is complete
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
        # Start memory processing in a background thread to avoid blocking the response
        threading.Thread(
            target=process_assistant_memory_background,
            args=(client, response_text, client.conversation_mode),
            daemon=True
        ).start()
        logger.info("Started assistant memory processing in background thread")

    # Mark first message as processed if this was the first message
    if not client.first_message_processed:
        client.first_message_processed = True
        logger.info("First message processed - future messages will use full memory context")

    return response_text

async def process_audio_file(client, audio_file_path=None):
    """Process an audio file and return a response."""
    if audio_file_path is None:
        raise ValueError("Audio file path cannot be None")

    logger.info(f"Processing audio file: {audio_file_path}")

    try:
        # Connect to the OpenAI WebSocket if not already connected
        if not client.ws or client.ws.closed:
            await client.connect()
            if not client.ws or client.ws.closed:
                logger.error("Failed to connect to OpenAI WebSocket")
                return "Failed to connect to voice service. Please try again."

        # Reset audio state
        client.reset_audio_state()

        # Read the audio file
        with open(audio_file_path, 'rb') as f:
            audio_data = f.read()

        # Base64 encode the audio data
        base64_audio = base64.b64encode(audio_data).decode('utf-8')

        # Send the audio data to the server
        await client.send_event({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        })

        # Commit the audio buffer
        await client.send_event({"type": "input_audio_buffer.commit"})

        # Create a response
        client.is_responding = True
        client.response_complete_event.clear()
        await client.send_event({"type": "response.create"})

        # Wait for the response to complete
        logger.info("Waiting for response to complete...")
        await client.response_complete_event.wait()

        # Get the response text
        response_text = client.current_response_text if hasattr(client, 'current_response_text') else "No response received"

        # Note: The audio is played automatically by the RealtimeClient when it receives
        # the response.audio.done event. We don't need to explicitly play it here.
        # The audio_handler.play_audio method is called in the event handler.

        # Add to transcript
        client.transcript_processor.add_assistant_response(response_text)

        # Process with memory if enabled
        if client.enable_memory and client.memory_integration:
            # Start memory processing in a background thread to avoid blocking the response
            threading.Thread(
                target=process_assistant_memory_background,
                args=(client, response_text, client.conversation_mode),
                daemon=True
            ).start()
            logger.info("Started assistant memory processing in background thread")

        # Mark first message as processed if this was the first message
        if not client.first_message_processed:
            client.first_message_processed = True
            logger.info("First message processed - future messages will use full memory context")

        return response_text

    except Exception as e:
        logger.error(f"Error processing audio file: {e}")
        return f"Error processing audio: {str(e)}"

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

def process_memory_background(client, message, conversation_mode):
    """Process memory operations in a background thread."""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process the message with memory integration
        loop.run_until_complete(
            client.memory_integration.process_user_message(message, conversation_mode)
        )

        # Close the loop
        loop.close()
        logger.info("Background memory processing completed")
    except Exception as e:
        logger.error(f"Error in background memory processing: {e}")

def process_assistant_memory_background(client, message, conversation_mode):
    """Process assistant memory operations in a background thread."""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process the message with memory integration
        loop.run_until_complete(
            client.memory_integration.process_assistant_message(message, conversation_mode)
        )

        # Close the loop
        loop.close()
        logger.info("Background assistant memory processing completed")
    except Exception as e:
        logger.error(f"Error in background assistant memory processing: {e}")

# Connect to Supabase
logger.info("Connecting to Supabase")
user_manager.connect()

# Check if we're running in the main process or the reloader process
def is_main_process():
    """Check if this is the main process or a reloader process."""
    import os

    # When Flask runs in debug mode, it sets this environment variable for the reloader process
    # WERKZEUG_RUN_MAIN is set to 'true' in the process that actually runs the application
    # and is not set in the first process that starts the reloader
    return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

# WebSocket event handlers for real-time audio streaming
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info(f"Client connected: {request.sid}")
    # Send a welcome message to confirm connection
    try:
        socketio.emit('connection_status', {'status': 'connected', 'message': 'Connected to server'}, room=request.sid)
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}")

@socketio.on('disconnect')
def handle_disconnect(reason=None):
    """Handle WebSocket disconnection.

    Args:
        reason: The reason for disconnection (provided by Socket.IO)
    """
    logger.info(f"Client disconnected: {request.sid}, reason: {reason}")

    # Clean up initialization timestamps
    if request.sid in stream_initialization_timestamps:
        del stream_initialization_timestamps[request.sid]
        logger.info(f"Removed initialization timestamp for {request.sid}")

    # Clean up session data
    if hasattr(handle_initialize_stream, 'user_sessions'):
        if request.sid in handle_initialize_stream.user_sessions:
            try:
                # Get the client to clean up
                session_data = handle_initialize_stream.user_sessions[request.sid]
                client = session_data.get('client')

                # Clean up client resources if needed
                # Instead of trying to close the WebSocket connection, just mark it for cleanup
                # The connection will be properly closed when the client reconnects or when the server restarts
                if client:
                    # Save sentiment results if enabled
                    if hasattr(client, 'enable_sentiment_analysis') and client.enable_sentiment_analysis:
                        if hasattr(client, 'sentiment_manager') and client.sentiment_manager:
                            try:
                                logger.info("Saving sentiment analysis results before disconnect...")

                                # Save sentiment results if we haven't already
                                if hasattr(client, 'sentiment_file_saved') and not client.sentiment_file_saved and hasattr(client, 'sentiment_file_path') and client.sentiment_file_path:
                                    # Save to the user-specific folder using our consistent file path
                                    client.sentiment_manager.save_results(custom_file_path=client.sentiment_file_path)
                                    logger.info(f"Saved sentiment analysis results to: {client.sentiment_file_path}")
                                    client.sentiment_file_saved = True
                                elif hasattr(client, 'sentiment_manager') and not hasattr(client, 'sentiment_file_path'):
                                    # Fall back to the default path
                                    client.sentiment_manager.save_results()
                                    if hasattr(client, 'sentiment_file_saved'):
                                        client.sentiment_file_saved = True
                            except Exception as e:
                                logger.error(f"Error saving sentiment results: {e}")

                    # Set a flag to indicate that the client is disconnected
                    client.is_disconnected = True
                    logger.info(f"Marked client as disconnected for {request.sid}")

                    # Log client state for debugging
                    ws_connected = hasattr(client, 'ws') and client.ws is not None
                    logger.info(f"Client WebSocket state: connected={ws_connected}, is_responding={getattr(client, 'is_responding', False)}")

                    # Don't remove the session, just mark it as disconnected
                    # This allows the client to reconnect later
                    logger.info(f"Keeping session data for possible reconnection: {request.sid}")
                else:
                    # If there's no client, we can safely remove the session
                    del handle_initialize_stream.user_sessions[request.sid]
                    logger.info(f"Removed session data for {request.sid} (no client)")
            except Exception as e:
                logger.error(f"Error during disconnect cleanup: {e}")
                import traceback
                logger.error(traceback.format_exc())

@socketio.on_error_default
def default_error_handler(e):
    """Handle all Socket.IO errors."""
    logger.error(f"Socket.IO error: {str(e)}")
    try:
        socketio.emit('error', {'message': f'Server error: {str(e)}'}, room=request.sid)
    except Exception as emit_error:
        logger.error(f"Error sending error message: {emit_error}")

@socketio.on('trigger_test_connection')
def handle_trigger_test_connection():
    """Handle a request to trigger a test connection after reconnection."""
    logger.info(f"Received trigger to test connection from {request.sid}")
    handle_test_openai_connection()

@socketio.on('test_openai_connection')
def handle_test_openai_connection():
    """Test the OpenAI WebSocket connection by sending a ping event."""
    try:
        logger.info(f"OpenAI connection test requested by {request.sid}")

        # Check if the client has a session
        if hasattr(handle_initialize_stream, 'user_sessions') and request.sid in handle_initialize_stream.user_sessions:
            session_data = handle_initialize_stream.user_sessions[request.sid]
            client = session_data.get('client')

            if client:
                # Check WebSocket connection
                ws_connected = hasattr(client, 'ws') and client.ws is not None
                is_disconnected = getattr(client, 'is_disconnected', False)

                # Log detailed connection state
                logger.info(f"WebSocket state for {request.sid}: has_ws={hasattr(client, 'ws')}, ws_not_none={client.ws is not None if hasattr(client, 'ws') else False}, is_disconnected={is_disconnected}")

                if ws_connected and not is_disconnected:
                    # Store the sid before starting the background task
                    sid = request.sid

                    # Emit immediate acknowledgment first
                    emit('openai_connection_test', {'status': 'testing'})

                    # Test the connection by checking if the WebSocket is connected
                    # We don't need to send a ping event, just check if the WebSocket is connected
                    try:
                        logger.info(f"Checking WebSocket connection for {sid}")

                        # Check if the WebSocket is connected
                        if hasattr(client, 'ws') and client.ws is not None and not getattr(client, 'is_disconnected', False):
                            # Check if the WebSocket is closed - use hasattr to check if the attribute exists
                            is_closed = False
                            if hasattr(client.ws, 'closed'):
                                is_closed = client.ws.closed
                            # For ClientConnection objects, check if they have a 'close' method instead
                            elif hasattr(client.ws, 'close') and callable(getattr(client.ws, 'close')):
                                # We can't directly check if it's closed, so we'll use other indicators
                                # For now, assume it's not closed if it has a close method
                                is_closed = False

                            if is_closed:
                                logger.warning(f"WebSocket is closed for {sid}")
                                socketio.emit('openai_connection_test', {
                                    'status': 'error',
                                    'message': 'WebSocket connection is closed'
                                }, room=sid)

                                # Try to reconnect
                                logger.info(f"Attempting to reconnect WebSocket for {sid}")

                                # Create a new event loop for this operation
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                                # Initialize audio components (this will connect to the WebSocket)
                                audio_success = loop.run_until_complete(client.initialize_audio())

                                # Close the loop
                                loop.close()

                                if audio_success:
                                    logger.info(f"Successfully reconnected WebSocket for {sid}")
                                    socketio.emit('openai_connection_test', {
                                        'status': 'success',
                                        'message': 'WebSocket connection has been reestablished'
                                    }, room=sid)
                                else:
                                    logger.error(f"Failed to reconnect WebSocket for {sid}")
                                    socketio.emit('openai_connection_test', {
                                        'status': 'error',
                                        'message': 'Failed to reconnect WebSocket'
                                    }, room=sid)
                            else:
                                logger.info(f"WebSocket connection is available for {sid}")
                                socketio.emit('openai_connection_test', {
                                    'status': 'success',
                                    'message': 'WebSocket connection is available'
                                }, room=sid)
                        else:
                            logger.warning(f"WebSocket connection is not available for {sid}")
                            socketio.emit('openai_connection_test', {
                                'status': 'error',
                                'message': 'WebSocket connection is not available'
                            }, room=sid)
                    except Exception as e:
                        logger.error(f"Error testing OpenAI connection: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        socketio.emit('openai_connection_test', {'status': 'error', 'message': str(e)}, room=sid)
                else:
                    logger.error(f"WebSocket not connected for {request.sid}")

                    # Try to reconnect the WebSocket
                    logger.info(f"Attempting to reconnect WebSocket for {request.sid}")

                    # Store the sid before starting the background task
                    sid = request.sid

                    # Use a named function that will be called immediately
                    def start_reconnect_websocket():
                        @socketio.start_background_task
                        def _reconnect_websocket():
                            try:
                                # Reset the disconnected flag
                                client.is_disconnected = False

                                # Create a new client instance instead of trying to reconnect the existing one
                                logger.info(f"Creating a new client instance for {sid}")

                                # Get the user ID and username from the session data
                                user_id = handle_initialize_stream.user_sessions[sid].get('user_id')
                                username = handle_initialize_stream.user_sessions[sid].get('username', 'Unknown User')

                                # Create a new client
                                new_client = get_or_create_client(user_id, username)
                                new_client.session_id = sid

                                # Replace the old client with the new one
                                handle_initialize_stream.user_sessions[sid]['client'] = new_client

                                logger.info(f"Successfully created new client for {sid}")
                                socketio.emit('openai_connection_test', {'status': 'reconnected'}, room=sid)

                                # Emit an event to trigger a new test
                                socketio.emit('trigger_test_connection', {}, room=sid)
                            except Exception as e:
                                logger.error(f"Error reconnecting WebSocket: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                socketio.emit('openai_connection_test', {'status': 'reconnection_error', 'message': str(e)}, room=sid)

                    # Start the reconnection process
                    start_reconnect_websocket()

                    # Emit immediate acknowledgment
                    emit('openai_connection_test', {'status': 'reconnecting'})
            else:
                logger.error(f"No client found for {request.sid}")
                emit('openai_connection_test', {'status': 'no_client'})
        else:
            logger.warning(f"No session found for {request.sid}, waiting for initialization")
            emit('openai_connection_test', {'status': 'waiting_for_initialization'})
    except Exception as e:
        logger.error(f"Error in handle_test_openai_connection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        emit('error', {'message': f'Server error: {str(e)}'})

@socketio.on('check_connection')
def handle_check_connection():
    """Check the WebSocket connection status."""
    try:
        logger.info(f"Connection check requested by {request.sid}")

        # Check if the client has a session
        if hasattr(handle_initialize_stream, 'user_sessions') and request.sid in handle_initialize_stream.user_sessions:
            session_data = handle_initialize_stream.user_sessions[request.sid]
            client = session_data.get('client')

            if client:
                # Check WebSocket connection
                ws_connected = hasattr(client, 'ws') and client.ws is not None
                is_disconnected = getattr(client, 'is_disconnected', False)
                is_responding = getattr(client, 'is_responding', False)

                # Log detailed connection state
                logger.info(f"WebSocket state for {request.sid}: has_ws={hasattr(client, 'ws')}, ws_not_none={client.ws is not None if hasattr(client, 'ws') else False}, is_disconnected={is_disconnected}, is_responding={is_responding}")

                # Emit connection status
                emit('connection_status', {
                    'status': 'connected' if ws_connected and not is_disconnected else 'disconnected',
                    'details': {
                        'ws_connected': ws_connected,
                        'is_disconnected': is_disconnected,
                        'is_responding': is_responding
                    }
                })

                # If disconnected, try to reconnect
                if not ws_connected or is_disconnected:
                    logger.info(f"Connection is not active, attempting to reconnect for {request.sid}")

                    # Reset the disconnected flag
                    client.is_disconnected = False

                    # Store the sid before starting the background task
                    sid = request.sid

                    # Use a named function that will be called immediately
                    def start_reconnect_client():
                        @socketio.start_background_task
                        def _reconnect_client():
                            try:
                                # Create a new client instance instead of trying to reconnect the existing one
                                logger.info(f"Creating a new client instance for {sid}")

                                # Get the user ID and username from the session data
                                user_id = handle_initialize_stream.user_sessions[sid].get('user_id')
                                username = handle_initialize_stream.user_sessions[sid].get('username', 'Unknown User')

                                # Create a new client
                                new_client = get_or_create_client(user_id, username)
                                new_client.session_id = sid

                                # Replace the old client with the new one
                                handle_initialize_stream.user_sessions[sid]['client'] = new_client

                                logger.info(f"Successfully created new client for {sid}")
                                socketio.emit('connection_status', {'status': 'reconnected'}, room=sid)
                            except Exception as e:
                                logger.error(f"Error reconnecting client: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                socketio.emit('error', {'message': f'Error reconnecting: {str(e)}'}, room=sid)

                    # Start the reconnection process
                    start_reconnect_client()
            else:
                logger.error(f"No client found for {request.sid}")
                emit('connection_status', {'status': 'no_client'})
        else:
            logger.warning(f"No session found for {request.sid}, waiting for initialization")
            emit('connection_status', {'status': 'waiting_for_initialization'})
    except Exception as e:
        logger.error(f"Error in handle_check_connection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        emit('error', {'message': f'Server error: {str(e)}'})

# Track initialization timestamps to prevent rapid duplicate requests
stream_initialization_timestamps = {}

@socketio.on('initialize_stream')
def handle_initialize_stream(data):
    """Initialize a streaming session for a user."""
    try:
        user_id = data.get('user_id', '')
        username = data.get('username', '')

        if not user_id:
            emit('error', {'message': 'User ID is required'})
            return

        # Check for duplicate initialization requests
        current_time = time.time()
        sid = request.sid

        # Check if we've recently processed an initialization request for this session
        if sid in stream_initialization_timestamps:
            last_init_time = stream_initialization_timestamps[sid]
            time_since_last_init = current_time - last_init_time

            # If it's been less than 5 seconds since the last initialization, ignore this request
            if time_since_last_init < 5:
                logger.info(f"Ignoring duplicate initialization request from {sid} (last request was {time_since_last_init:.2f} seconds ago)")
                # Still emit success to prevent client from retrying
                emit('stream_initialized', {'status': 'success', 'message': 'Duplicate request ignored'})
                return

        # Update the initialization timestamp for this session
        stream_initialization_timestamps[sid] = current_time

        logger.info(f"Initializing streaming session for user {username} (ID: {user_id})")

        # Store the user ID in a global dictionary instead of the session
        # This allows us to emit events to this specific client
        if not hasattr(handle_initialize_stream, 'user_sessions'):
            handle_initialize_stream.user_sessions = {}

        # Check if there's already a session for this client
        if request.sid in handle_initialize_stream.user_sessions:
            logger.info(f"Session already exists for {request.sid}, checking connection status")
            existing_client = handle_initialize_stream.user_sessions[request.sid].get('client')

            if existing_client:
                # Check WebSocket connection
                ws_connected = hasattr(existing_client, 'ws') and existing_client.ws is not None
                is_disconnected = getattr(existing_client, 'is_disconnected', False)

                # Log detailed connection state
                logger.info(f"WebSocket state for {request.sid}: has_ws={hasattr(existing_client, 'ws')}, ws_not_none={existing_client.ws is not None if hasattr(existing_client, 'ws') else False}, is_disconnected={is_disconnected}")

                # Reset the disconnected flag if it was set
                existing_client.is_disconnected = False
                logger.info(f"Reset disconnected flag for existing client")

                # Update the session ID
                existing_client.session_id = request.sid

                # Emit success and return
                emit('stream_initialized', {'status': 'success', 'message': 'Session reused'})
                return

        # Create a new session
        handle_initialize_stream.user_sessions[request.sid] = {
            'user_id': user_id,
            'client': None,
            'last_init_time': current_time  # Track when this session was initialized
        }

        # Get or create client for this user
        client = get_or_create_client(user_id, username)

        # Store the client in our dictionary
        handle_initialize_stream.user_sessions[request.sid]['client'] = client

        # Also store the session ID with the client for emitting events
        client.session_id = request.sid
        client.is_disconnected = False  # Reset disconnected flag

        # Store the sid before starting the background task
        sid = request.sid

        # Connect the client to the OpenAI WebSocket synchronously
        # This ensures the connection is established before we return success
        try:
            # Check if the client is already connected
            ws_connected = hasattr(client, 'ws') and client.ws is not None

            if not ws_connected:
                logger.info(f"Connecting client to OpenAI WebSocket synchronously")

                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Initialize audio components (this will connect to the WebSocket)
                audio_success = loop.run_until_complete(client.initialize_audio())

                # Close the loop
                loop.close()

                if audio_success:
                    logger.info(f"Successfully connected client to OpenAI WebSocket")
                    # Emit success after the connection is established
                    emit('stream_initialized', {'status': 'success'})
                else:
                    logger.error(f"Failed to connect client to OpenAI WebSocket")
                    emit('error', {'message': 'Failed to connect to voice service. Please try again.'})
                    return
            else:
                logger.info(f"Client already connected to OpenAI WebSocket")
                # Emit success since the client is already connected
                emit('stream_initialized', {'status': 'success'})
        except Exception as e:
            logger.error(f"Error connecting client to OpenAI WebSocket: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                emit('error', {'message': f'Error connecting to voice service: {str(e)}'})
            except Exception as emit_error:
                logger.error(f"Error sending error message: {emit_error}")

    except Exception as e:
        logger.error(f"Error in handle_initialize_stream: {e}")
        try:
            emit('error', {'message': f'Server error: {str(e)}'})
        except Exception as emit_error:
            logger.error(f"Error sending error message: {emit_error}")

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunks from the client."""
    try:
        # Log the receipt of audio chunk
        logger.debug(f"Received audio chunk from {request.sid}")

        # Get session data from our dictionary
        if not hasattr(handle_initialize_stream, 'user_sessions'):
            logger.error("No user sessions dictionary found")
            emit('error', {'message': 'No sessions initialized'})
            return

        session_data = handle_initialize_stream.user_sessions.get(request.sid)
        if not session_data:
            logger.error(f"No session data found for {request.sid}")
            emit('error', {'message': 'Session not initialized'})
            return

        client = session_data.get('client')
        if not client:
            logger.error(f"No client found in session data for {request.sid}")
            emit('error', {'message': 'Client not initialized'})
            return

        # Get the audio chunk from the data
        audio_chunk = data.get('audio')
        if not audio_chunk:
            logger.error("No audio data provided in chunk")
            emit('error', {'message': 'No audio data provided'})
            return

        # Log audio chunk size for debugging
        logger.debug(f"Audio chunk size: {len(audio_chunk)} bytes")

        # Check if client has a WebSocket connection
        ws_connected = hasattr(client, 'ws') and client.ws is not None
        # Check for closed attribute or close method
        if ws_connected:
            is_closed = False
            if hasattr(client.ws, 'closed'):
                is_closed = client.ws.closed
            # For ClientConnection objects, check if they have a 'close' method instead
            elif hasattr(client.ws, 'close') and callable(getattr(client.ws, 'close')):
                # We can't directly check if it's closed, so we'll use other indicators
                # For now, assume it's not closed if it has a close method
                is_closed = False
            ws_connected = not is_closed

        # Also check the disconnected flag
        is_disconnected = getattr(client, 'is_disconnected', False)

        if not ws_connected or is_disconnected:
            logger.error(f"Client WebSocket not connected for {request.sid}")
            # Log detailed connection state
            logger.error(f"WebSocket state: has_ws={hasattr(client, 'ws')}, ws_not_none={client.ws is not None if hasattr(client, 'ws') else False}, is_disconnected={is_disconnected}")

            # Try to reconnect the WebSocket
            logger.info("Attempting to reconnect WebSocket...")

            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Reset the disconnected flag
                client.is_disconnected = False

                # Initialize audio components (this will connect to the WebSocket)
                reconnected = loop.run_until_complete(client.initialize_audio())

                # Close the loop
                loop.close()

                if reconnected:
                    logger.info("Successfully reconnected WebSocket")
                    # Don't emit an error, just continue with the audio chunk
                else:
                    logger.error("Failed to reconnect WebSocket")
                    emit('error', {'message': 'Voice service not connected and reconnection failed'})
                    return
            except Exception as e:
                logger.error(f"Error reconnecting WebSocket: {e}")
                emit('error', {'message': f'Error reconnecting to voice service: {str(e)}'})
                return

        # Process the audio chunk
        try:
            # Decode the base64 audio chunk
            audio_data = base64.b64decode(audio_chunk)

            # Check for silence and update the shared emotion state
            client._check_for_silence(audio_data)

            # Queue the audio chunk to be sent asynchronously
            # This avoids the event loop issues
            if not hasattr(client, 'audio_queue'):
                client.audio_queue = []

            # Add the chunk to the queue
            client.audio_queue.append(audio_chunk)

            # Start a background task to process the queue if not already running
            if not hasattr(client, 'is_processing_queue') or not client.is_processing_queue:
                client.is_processing_queue = True

                # Define and start the background task in one step
                socketio.start_background_task(
                    process_audio_queue_task,
                    client=client,
                    logger=logger
                )

            # For real-time conversation, we want to ensure the server-side VAD is working
            # We'll track the time since the last commit to ensure we're processing audio regularly
            if not hasattr(client, 'last_commit_time'):
                client.last_commit_time = 0

            # Auto-commit if it's been more than 1 second since the last commit
            # This ensures the server processes audio even if the client doesn't explicitly commit
            current_time = time.time()
            if current_time - getattr(client, 'last_commit_time', 0) > 1.0:
                # We'll do this in a background task to avoid blocking
                socketio.start_background_task(
                    auto_commit_audio,
                    client=client,
                    sid=request.sid,
                    logger=logger
                )

            # Log audio activity for debugging
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            max_amp = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
            if max_amp > 1000:  # Only log significant audio activity
                logger.info(f"Audio activity detected with amplitude: {max_amp}")
                # Send a debug event to the client to confirm audio is being received
                socketio.emit('audio_debug', {
                    'message': f'Audio received with amplitude: {max_amp}',
                    'timestamp': time.time()
                }, room=request.sid)

                # Send audio chunk to sentiment analysis if enabled
                if client.enable_sentiment_analysis and hasattr(client, 'sentiment_manager') and client.sentiment_manager:
                    try:
                        # Send the audio chunk for sentiment analysis
                        client.sentiment_manager.send_audio_chunk(audio_data, 16000, "user")  # Assuming 16kHz sample rate
                        logger.debug("Sent audio chunk to sentiment analysis")
                    except Exception as e:
                        logger.error(f"Error sending audio chunk to sentiment analysis: {e}")

            # Update the last activity time
            client.last_audio_activity_time = time.time()

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            logger.error(traceback.format_exc())
            emit('error', {'message': f'Error processing audio: {str(e)}'})

    except Exception as e:
        logger.error(f"Unexpected error in handle_audio_chunk: {e}")
        emit('error', {'message': f'Server error: {str(e)}'})

# Define the background task function outside the handler
def process_audio_queue_task(client, logger):
    try:
        # Process all chunks in the queue
        while hasattr(client, 'audio_queue') and client.audio_queue:
            # Get the next chunk
            chunk = client.audio_queue.pop(0)

            # Send the audio chunk to the OpenAI WebSocket
            if hasattr(client, 'ws') and client.ws is not None and not getattr(client, 'is_disconnected', False):
                try:
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Send the event using the client's send_event method
                    loop.run_until_complete(client.send_event({
                        "type": "input_audio_buffer.append",
                        "audio": chunk
                    }))

                    # Close the loop
                    loop.close()

                    logger.debug("Sent audio chunk to OpenAI WebSocket")
                except Exception as e:
                    logger.error(f"Error sending audio chunk: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # Reset the processing flag
        client.is_processing_queue = False
    except Exception as e:
        logger.error(f"Error processing audio queue: {e}")
        client.is_processing_queue = False

# Function to automatically commit audio for real-time conversation
def auto_commit_audio(client, sid, logger):
    try:
        # Only proceed if the client is not already responding
        if getattr(client, 'is_responding', False):
            logger.debug("Client is already responding, skipping auto-commit")
            return

        # Update the last commit time
        client.last_commit_time = time.time()

        # Send the commit event using an event loop
        if hasattr(client, 'ws') and client.ws is not None and not getattr(client, 'is_disconnected', False):
            try:
                # Create a new event loop for this operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Send the commit event using the client's send_event method
                loop.run_until_complete(client.send_event({
                    "type": "input_audio_buffer.commit"
                }))
                logger.info(f"Auto-committed audio buffer for {sid}")

                # Add a small delay to ensure the commit is processed
                time.sleep(0.2)

                # With server-side VAD, response.create should happen automatically
                # But we'll explicitly create a response as a fallback
                if not getattr(client, 'is_responding', False):
                    # Set responding state
                    client.is_responding = True
                    if hasattr(client, 'response_complete_event'):
                        client.response_complete_event.clear()

                    # Send the response.create event
                    loop.run_until_complete(client.send_event({
                        "type": "response.create"
                    }))
                    logger.info(f"Response creation initiated after auto-commit for {sid}")

                    # Emit a debug event to help troubleshoot
                    socketio.emit('debug', {'message': 'Response creation initiated after auto-commit'}, room=sid)

                # Close the loop
                loop.close()
            except Exception as e:
                logger.error(f"Error in auto-commit: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error in auto_commit_audio: {e}")
        import traceback
        logger.error(traceback.format_exc())

@socketio.on('force_response')
def handle_force_response():
    """Force a response from the assistant for debugging purposes."""
    try:
        logger.info(f"Force response requested by {request.sid}")

        # Get session data
        if not hasattr(handle_initialize_stream, 'user_sessions'):
            logger.error("No user sessions dictionary found")
            emit('error', {'message': 'No sessions initialized'})
            return

        session_data = handle_initialize_stream.user_sessions.get(request.sid)
        if not session_data:
            logger.error(f"No session data found for {request.sid}")
            emit('error', {'message': 'Session not initialized'})
            return

        client = session_data.get('client')
        if not client:
            logger.error(f"No client found in session data for {request.sid}")
            emit('error', {'message': 'Client not initialized'})
            return

        # Check if client has a WebSocket connection
        if not hasattr(client, 'ws') or not client.ws or getattr(client, 'is_disconnected', False):
            logger.error(f"Client WebSocket not connected for {request.sid}")
            emit('error', {'message': 'Voice service not connected'})
            return

        # Store the sid
        sid = request.sid

        # Create a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Send a test message to force a response
            loop.run_until_complete(client.send_event({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(b"test audio data").decode('utf-8')
            }))

            # Commit the audio buffer
            loop.run_until_complete(client.send_event({
                "type": "input_audio_buffer.commit"
            }))

            # Create a response
            if not client.is_responding:
                client.is_responding = True
                if hasattr(client, 'response_complete_event'):
                    client.response_complete_event.clear()

                # Send the response.create event
                loop.run_until_complete(client.send_event({
                    "type": "response.create"
                }))

                logger.info(f"Forced response creation for {sid}")
                socketio.emit('debug', {'message': 'Forced response creation'}, room=sid)
            else:
                logger.warning(f"Client is already responding for {sid}")
                socketio.emit('debug', {'message': 'Client is already responding'}, room=sid)

            # Close the loop
            loop.close()

            emit('debug', {'status': 'success', 'message': 'Force response command sent'})
        except Exception as e:
            logger.error(f"Error forcing response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            emit('error', {'message': f'Error forcing response: {str(e)}'})

            # Close the loop
            loop.close()
    except Exception as e:
        logger.error(f"Unexpected error in handle_force_response: {e}")
        emit('error', {'message': f'Server error: {str(e)}'})

@socketio.on('commit_audio')
def handle_commit_audio():
    """Commit the audio buffer and request a response."""
    try:
        # Get session data from our dictionary
        if not hasattr(handle_initialize_stream, 'user_sessions'):
            logger.error("No user sessions dictionary found")
            emit('error', {'message': 'No sessions initialized'})
            return

        session_data = handle_initialize_stream.user_sessions.get(request.sid)
        if not session_data:
            logger.error(f"No session data found for {request.sid}")
            emit('error', {'message': 'Session not initialized'})
            return

        client = session_data.get('client')
        if not client:
            logger.error(f"No client found in session data for {request.sid}")
            emit('error', {'message': 'Client not initialized'})
            return

        # Check if client has a WebSocket connection
        if not hasattr(client, 'ws') or not client.ws or getattr(client, 'is_disconnected', False):
            logger.error(f"Client WebSocket not connected for {request.sid}")
            emit('error', {'message': 'Voice service not connected'})
            return

        logger.info(f"Committing audio buffer for {request.sid}")

        # Update the last commit time to prevent auto-commits right after an explicit commit
        if hasattr(client, 'last_commit_time'):
            client.last_commit_time = time.time()

        # Store the sid before starting the background task
        sid = request.sid

        # Commit the audio buffer and create a response directly
        try:
            # Send the commit event using an event loop
            if hasattr(client, 'ws') and client.ws is not None and not getattr(client, 'is_disconnected', False):
                try:
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Send the commit event using the client's send_event method
                    loop.run_until_complete(client.send_event({
                        "type": "input_audio_buffer.commit"
                    }))
                    logger.info(f"Audio buffer committed for {sid}")

                    # Add a small delay to ensure the commit is processed
                    time.sleep(0.2)

                    # Create a response if not already responding
                    # Note: With server-side VAD, this should happen automatically,
                    # but we'll do it explicitly as a fallback
                    if not client.is_responding:
                        client.is_responding = True
                        if hasattr(client, 'response_complete_event'):
                            client.response_complete_event.clear()

                        # Send the response.create event using the client's send_event method
                        loop.run_until_complete(client.send_event({
                            "type": "response.create"
                        }))
                        logger.info(f"Response creation initiated for {sid}")

                        # Emit an event to the client to indicate that the response is being generated
                        socketio.emit('response_started', {'status': 'generating'}, room=sid)

                        # Also emit a debug event to help troubleshoot
                        socketio.emit('debug', {'message': 'Response creation initiated'}, room=sid)
                    else:
                        logger.warning(f"Client is already responding for {sid}, skipping response.create")
                        # Emit a debug event to help troubleshoot
                        socketio.emit('debug', {'message': 'Client is already responding, skipping response.create'}, room=sid)

                    # Close the loop
                    loop.close()

                    logger.info(f"Audio buffer committed and response created for {sid}")
                except Exception as e:
                    logger.error(f"Error sending commit/response events: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Reset responding state on error
                    client.is_responding = False
                    if hasattr(client, 'response_complete_event'):
                        client.response_complete_event.set()
                    socketio.emit('error', {'message': f'Error processing audio: {str(e)}'}, room=sid)
            else:
                logger.error(f"WebSocket not connected for {sid}")
                socketio.emit('error', {'message': 'Voice service not connected'}, room=sid)
        except Exception as e:
            logger.error(f"Error committing audio buffer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Reset responding state on error
            client.is_responding = False
            if hasattr(client, 'response_complete_event'):
                client.response_complete_event.set()
            socketio.emit('error', {'message': f'Error processing audio: {str(e)}'}, room=sid)

        emit('audio_committed', {'status': 'success'})
    except Exception as e:
        logger.error(f"Unexpected error in handle_commit_audio: {e}")
        emit('error', {'message': f'Server error: {str(e)}'})

# Override the handle_event method of RealtimeClient to emit WebSocket events
def emit_realtime_events(client, event):
    """Emit events from the RealtimeClient to the WebSocket client."""
    try:
        event_type = event.get('type')

        # Log all events for debugging
        logger.debug(f"Received event from OpenAI: {event_type}")

        # Log full event details for important events
        important_events = ["error", "response.started", "response.done", "transcript.final"]
        if event_type in important_events:
            logger.info(f"Important event details: {event}")

        # Get the session ID from the client
        session_id = getattr(client, 'session_id', None)
        if not session_id:
            logger.error(f"No session ID found for client, can't emit event: {event_type}")
            return

        # Check if the client is marked as disconnected
        if getattr(client, 'is_disconnected', False):
            logger.warning(f"Client is marked as disconnected, skipping event: {event_type}")
            return

        # Handle different event types
        if event_type == "response.text.delta":
            # Text response delta
            delta = event.get('delta', '')
            if delta:
                logger.info(f"Emitting text delta: {delta[:20]}...")
                # Emit both event names to ensure compatibility
                socketio.emit('text_response', {'delta': delta}, room=session_id)
                socketio.emit('text_response_delta', {'delta': delta}, room=session_id)
                logger.info(f"Text response emitted to session {session_id}")

        elif event_type == "response.audio.delta":
            # Audio response delta
            audio_data = event.get('delta', '')
            if audio_data:
                logger.info(f"Emitting audio delta of length {len(audio_data)}")
                # Emit both event names to ensure compatibility
                socketio.emit('audio_response', {'delta': audio_data}, room=session_id)
                socketio.emit('audio_response_delta', {'delta': audio_data}, room=session_id)
                logger.info(f"Audio response emitted to session {session_id}")

        elif event_type == "response.done":
            # Response complete
            logger.info(f"Response complete for session {session_id}")
            # Reset the responding state
            client.is_responding = False
            if hasattr(client, 'response_complete_event'):
                client.response_complete_event.set()

            # Process sentiment analysis for the assistant's response
            if client.enable_sentiment_analysis and hasattr(client, 'sentiment_manager') and client.sentiment_manager:
                try:
                    # Set assistant sentiment scores
                    client._set_assistant_sentiment_scores()
                    logger.info("Updated assistant sentiment scores")
                except Exception as e:
                    logger.error(f"Error updating assistant sentiment scores: {e}")

            socketio.emit('response_complete', {'status': 'success'}, room=session_id)

        elif event_type == "response.started":
            # Response started
            logger.info(f"Response started for session {session_id}")
            socketio.emit('response_started', {'status': 'generating'}, room=session_id)

        elif event_type == "response.interrupted":
            # Response interrupted
            logger.info(f"Response interrupted for session {session_id}")
            # Reset the responding state
            client.is_responding = False
            if hasattr(client, 'response_complete_event'):
                client.response_complete_event.set()
            socketio.emit('response_interrupted', {'status': 'interrupted'}, room=session_id)

        elif event_type == "turn.start":
            # User turn started (speech detected)
            logger.info(f"User turn started for session {session_id}")
            socketio.emit('turn_started', {'status': 'speaking'}, room=session_id)

        elif event_type == "turn.end":
            # User turn ended (silence detected)
            logger.info(f"User turn ended for session {session_id}")
            socketio.emit('turn_ended', {'status': 'silence'}, room=session_id)

        elif event_type == "transcript.partial":
            # Partial transcript
            transcript = event.get('transcript', '')
            if transcript:
                logger.debug(f"Emitting partial transcript: {transcript[:30]}...")
                socketio.emit('transcript_partial', {'transcript': transcript}, room=session_id)

        elif event_type == "transcript.final":
            # Final transcript
            transcript = event.get('transcript', '')
            if transcript:
                logger.info(f"Emitting final transcript: {transcript[:30]}...")
                socketio.emit('transcript_final', {'transcript': transcript}, room=session_id)

        elif event_type == "error":
            # Error event
            error_msg = event.get('error', {}).get('message', 'Unknown error')
            logger.error(f"Error from OpenAI: {error_msg}")
            # Reset the responding state on error
            client.is_responding = False
            if hasattr(client, 'response_complete_event'):
                client.response_complete_event.set()
            socketio.emit('error', {'message': error_msg}, room=session_id)

        else:
            # Other event types
            logger.debug(f"Received unhandled event type: {event_type}")
    except Exception as e:
        logger.error(f"Error in emit_realtime_events: {e}")
        # Try to send an error message to the client if possible
        try:
            if 'session_id' in locals() and session_id:
                socketio.emit('error', {'message': f'Server error: {str(e)}'}, room=session_id)
        except:
            pass

if __name__ == '__main__':
    # Only log in the main process to avoid duplicate logs
    if is_main_process():
        logger.info("Starting API server in main process")
    else:
        logger.info("Starting API server in reloader process")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
