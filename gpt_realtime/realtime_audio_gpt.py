import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
# Import logging configuration
from gpt_realtime.logging_config import configure_logging
configure_logging()

import asyncio
import websockets
import json
import base64
import logging
import time
import numpy as np
import multiprocessing
import queue
import threading
import random

import traceback

from gpt_realtime.transcript_and_audio import AudioHandler, TranscriptProcessor


import openai

# Dash / Plotting Imports
import dash
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output

# Import memory integration
from memory.memory_integration import MemoryIntegration

# Import user account manager
from accounts.user_account_manager import UserAccountManager

# Attempt to import sentiment analyzer, handle gracefully if it doesn't exist
try:
    from gpt_realtime.sentiment_analyzer import SentimentAnalysisProcess
except ImportError:
    SentimentAnalysisProcess = None
    print("WARNING: sentiment_analyzer.py not found or SentimentAnalysisProcess class missing.")
    print("Sentiment analysis feature will be disabled.")

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

# Helper function to format data for RadarChart
def format_data_for_radar(emotion_scores):
    """Converts {'Emotion': score} dict to [{'emotion': Emotion, 'Sentiment': score}] list."""
    # Ensure all core emotions are present, defaulting to 0 if missing
    formatted = []
    for emotion in CORE_EMOTIONS:
        score = emotion_scores.get(emotion, 0) # Default to 0 if emotion not in scores
        formatted.append({"emotion": emotion, "Sentiment": score})
    return formatted

logger = logging.getLogger(__name__)

class RealtimeClient:
    """Client for interacting with the OpenAI Realtime API via WebSocket."""
    def __init__(self, api_key, google_api_key=None, voice="alloy", enable_sentiment_analysis=False, shared_emotion_scores=None, conversation_mode="audio", enable_memory=True, user_id=None, username=None):
        # WebSocket Configuration
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-mini-realtime-preview-2024-12-17"
        self.chat_model = "gpt-4o"  # Model for chat completions
        self.api_key = api_key # OpenAI API Key
        self.google_api_key = google_api_key
        self.voice = voice
        self.ws = None
        self.audio_handler = AudioHandler()

        # User information
        self.user_id = user_id
        self.username = username

        # Initialize OpenAI client for chat completions
        self.openai_client = openai.OpenAI(api_key=api_key)

        self.audio_buffer = b''  # Buffer for streaming audio responses

        # Add flag to track if assistant is currently responding
        self.is_responding = False
        self.response_complete_event = asyncio.Event()
        self.response_complete_event.set()  # Initially set to True (not responding)

        # Create transcript processor
        self.transcript_processor = TranscriptProcessor()

        # Flag to track if we received a transcript for current audio session
        self.received_transcript = False

        # Conversation mode: "audio" or "chat"
        self.conversation_mode = conversation_mode

        # Chat history for text-based conversations
        self.chat_history = []

        # Memory integration
        self.enable_memory = enable_memory
        self.memory_integration = None

        # Memory context flags
        self.waiting_to_add_memory = False
        self.memory_context_to_add = None

        # Shutdown flag to prevent processing events after quit
        self.shutdown_in_progress = False

        if self.enable_memory:
            try:
                # Load Supabase credentials from .api_key.json
                try:
                    with open('.api_key.json', 'r') as f:
                        api_keys = json.load(f)
                        supabase_url = api_keys.get('supabase_url')
                        supabase_key = api_keys.get('supabase_api_key')
                except Exception as e:
                    logger.warning(f"Could not load Supabase credentials: {e}")
                    supabase_url = None
                    supabase_key = None

                self.memory_integration = MemoryIntegration(
                    openai_api_key=api_key,
                    gemini_api_key=google_api_key,
                    user_id=user_id,
                    supabase_url=supabase_url,
                    supabase_key=supabase_key
                )
                logger.info(f"Memory integration enabled for user: {username if username else 'Anonymous'}")
            except Exception as e:
                logger.error(f"Failed to initialize memory integration: {e}")
                self.enable_memory = False

        # Server-side VAD configuration
        self.VAD_turn_detection = True
        self.VAD_config = {
            "type": "server_vad",
            "threshold": 0.3,  # Lower threshold to make it more sensitive (0.5 -> 0.3)
            "prefix_padding_ms": 500,  # Increase padding before speech (300 -> 500)
            "silence_duration_ms": 1000,  # Increase silence duration to detect end of speech (600 -> 1000)
            "create_response": True,  # Explicitly enable auto-response creation
            "interrupt_response": True  # Enable interrupting the model's response
        }

        # Session configuration
        # Create personalized instructions if username is available
        user_greeting = f"The user's name is {self.username}." if self.username else ""

        self.session_config = {
            "modalities": ["audio", "text"],
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "instructions": f"You are a helpful AI assistant with access to memories from previous conversations. {user_greeting} These memories will be provided to you as system messages starting with '### Relevant memories from previous conversations:'. Use these memories to provide personalized and contextually relevant responses. When appropriate, reference these past interactions naturally. Respond in a clear and engaging way. If the user's audio is quiet or unclear, still do your best to respond appropriately.\n\nYou have the ability to create new memories about the user. When you want to explicitly remember something important about the user, you can use phrases like 'I'll remember that about you' or 'I'll make a note of that.' When you use these phrases, the system will automatically store this information in your memory. This is particularly useful for important user preferences, personal details, or significant events that would be helpful to recall in future conversations.\n\nYou can also explicitly trigger memory retrieval by using phrases like 'let me recall' or 'let me see what I know about you' or 'based on our previous conversations'. When you use these phrases, the system will prioritize retrieving relevant memories to help you provide more personalized responses.\n\nIf you attempt to recall information but no relevant memories are found, you will receive a system message starting with '### Memory retrieval note:'. In such cases, acknowledge that you don't have that specific information in your memory and respond based on your general knowledge instead. Be honest about what you don't know about the user's past interactions.",
            "turn_detection": self.VAD_config if self.VAD_turn_detection else None,
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "temperature": 0.7
        }

        # --- Sentiment Analysis Setup ---
        self.google_api_key = google_api_key
        self.enable_sentiment_analysis = enable_sentiment_analysis and SentimentAnalysisProcess is not None
        self.sentiment_model_name = "models/gemini-2.0-flash"
        self.audio_mp_queue = None
        self.sentiment_history_list = None
        self.sentiment_process = None
        self.manager = None
        self.sentiment_update_thread = None
        self.sentiment_stop_event = threading.Event()
        self.shared_emotion_scores = shared_emotion_scores

        if self.enable_sentiment_analysis:
            if not self.google_api_key:
                logger.warning("Sentiment analysis enabled, but Google API key is missing. Disabling.")
                self.enable_sentiment_analysis = False
            else:
                try:
                    self.manager = multiprocessing.Manager()
                    self.audio_mp_queue = self.manager.Queue(maxsize=500) # Queue for audio chunks
                    self.sentiment_history_list = self.manager.list() # Shared list for results
                    logger.info(f"Sentiment analysis components initialized for model {self.sentiment_model_name}.")
                except Exception as e:
                    logger.error(f"Failed to initialize multiprocessing components for sentiment analysis: {e}")
                    self.enable_sentiment_analysis = False
                    # Clean up partial initialization
                    if self.manager:
                         try: self.manager.shutdown()
                         except Exception: pass
                    self.manager = None
                    self.audio_mp_queue = None
                    self.sentiment_history_list = None
        else:
             if enable_sentiment_analysis: # Log if it was requested but couldn't be enabled
                 logger.warning("Sentiment analysis was requested but could not be enabled (check imports/API key).")

    async def connect(self):
        """Connect to the WebSocket server."""
        # logger.info(f"Connecting to WebSocket: {self.url}")

        # Start the transcript processor
        self.transcript_processor.start()

        # Reset WebSocket connection if it exists
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

        try:
            # Create the URL with the model parameter in the query string
            full_url = f"{self.url}?model={self.model}"

            # Prepare headers dictionary
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            # Connect to the WebSocket
            try:
                self.ws = await websockets.connect(
                    full_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=60
                )
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                return False

            # logger.info("Successfully connected to OpenAI Realtime API")

            # Configure session
            try:
                await self.send_event({
                    "type": "session.update",
                    "session": self.session_config
                })
                # logger.info("Session configuration sent")
            except Exception as e:
                logger.error(f"Failed to send session configuration: {e}")
                await self.cleanup()
                return False

            # Wait for acknowledgment
            try:
                session_ack = await asyncio.wait_for(self.ws.recv(), timeout=5)
                session_ack_json = json.loads(session_ack)
                if session_ack_json.get("type") == "session.updated":
                     logger.info("Session successfully initialized")
                else:
                    logger.warning(f"Unexpected session response: {session_ack_json}")
            except asyncio.TimeoutError:
                logger.warning("No session acknowledgment received within timeout")
            except Exception as e:
                logger.error(f"Error receiving session acknowledgment: {e}")
                # Continue anyway as some implementations might not send an acknowledgment

            # --- Start Sentiment Analysis Process (if enabled) ---
            if self.enable_sentiment_analysis and self.sentiment_process is None: # Start only if enabled and not already running
                logger.info("Starting sentiment analysis process...")
                try:
                    # Ensure components are valid before starting
                    if not self.google_api_key or not self.audio_mp_queue or self.sentiment_history_list is None or not SentimentAnalysisProcess or self.shared_emotion_scores is None:
                         raise ValueError("Missing required components for sentiment analysis or shared dict.")

                    self.sentiment_process = SentimentAnalysisProcess(
                        api_key=self.google_api_key,
                        sentiment_model_name=self.sentiment_model_name,
                        audio_queue=self.audio_mp_queue,
                        sentiment_history_list=self.sentiment_history_list
                    )
                    self.sentiment_process.start()
                    logger.info("Sentiment analysis process started successfully.")

                    # Start the sentiment update thread
                    self.sentiment_stop_event.clear()
                    self.sentiment_update_thread = threading.Thread(
                        target=self._continuously_update_sentiment, daemon=True
                    )
                    self.sentiment_update_thread.start()
                    logger.info("Sentiment update thread started.")

                except Exception as e:
                    logger.error(f"Failed to start sentiment analysis process or update thread: {e}")
                    # Ensure process is None if start fails, and disable for this run
                    self.sentiment_process = None
                    self.enable_sentiment_analysis = False # Disable if it fails to start
                    logger.warning("Sentiment analysis has been disabled due to startup failure.")

            # Make sure we're not already in responding state
            self.is_responding = False
            self.response_complete_event.set()

            # Send initial response.create - but only if we're not already in a response
            try:
                self.is_responding = True
                self.response_complete_event.clear()
                await self.send_event({"type": "response.create"})
            except Exception as e:
                logger.error(f"Failed to send initial response.create: {e}")
                await self.cleanup()
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            logger.error(f"Connection error details: {str(e)}")
            if self.ws:
                try:
                    await self.ws.close()
                except:
                    pass
                self.ws = None
            return False

    async def send_event(self, event):
        """Send an event to the WebSocket server."""
        if self.ws:
            try:
                await self.ws.send(json.dumps(event))
            except websockets.exceptions.ConnectionClosedError as e:
                logger.error(f"Connection closed while sending event: {e}")
                # Try to reconnect
                # logger.info("Attempting to reconnect...")
                reconnected = await self.connect()
                if reconnected:
                    # logger.info("Successfully reconnected, retrying event send")
                    # Try sending the event again
                    await self.ws.send(json.dumps(event))
                else:
                    logger.error("Failed to reconnect")
                    raise RuntimeError("Failed to reconnect to WebSocket server")
            except Exception as e:
                logger.error(f"Error sending event: {e}")
                # Re-raise to allow proper error handling by caller
                raise
        else:
            logger.error("Cannot send event: WebSocket is not connected")
            raise RuntimeError("WebSocket connection is not established")

    async def receive_events(self):
        """Continuously receive events from the WebSocket server."""
        # In chat mode, we don't need to receive events from the WebSocket
        if self.conversation_mode == "chat":
            return

        if not self.ws:
            logger.error("Cannot receive events: WebSocket is not connected")
            return

        try:
            # Use async for with the WebSocket connection
            async for message in self.ws:
                try:
                    event = json.loads(message)
                    await self.handle_event(event)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except Exception as e:
            # Generic exception catch for all WebSocket-related errors
            logger.error(f"WebSocket connection error: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Update the connection status
            self.ws = None

    async def handle_event(self, event):
        """Handle incoming events from the WebSocket server."""
        # If shutdown is in progress, ignore all events
        if hasattr(self, 'shutdown_in_progress') and self.shutdown_in_progress:
            return

        event_type = event.get("type")

        # Log all event types to help with debugging
        # logger.debug(f"Received event type: {event_type}")

        # Special handling for transcript events
        if event_type.startswith("input_audio_transcript") or event_type == "conversation.item.input_audio_transcription.completed":
            # logger.info(f"Speech transcript event: {event_type}")
            # logger.debug(f"Event data: {event}")

            # Flag that we received a transcript
            self.received_transcript = True

            # For conversation.item.input_audio_transcription.completed events
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = None
                # According to documentation, transcript may be in different locations
                if "transcript" in event:
                    transcript = event.get("transcript", "")
                elif "item" in event:
                    item = event.get("item", {})
                    if "content" in item:
                        for content in item["content"]:
                            if content.get("type") == "input_audio" and "transcript" in content:
                                transcript = content.get("transcript", "")
                                break

                if transcript:
                    print(f"\nTranscribed: {transcript}")
                    # Add to transcript
                    self.transcript_processor.add_user_query(transcript)

                    # Flag to track if we need to create a response
                    create_response = True

                    # Process with memory agent if enabled
                    if self.enable_memory and self.memory_integration:
                        try:
                            # Process user message with memory agent
                            asyncio.create_task(self.memory_integration.process_user_message(transcript, self.conversation_mode))
                            logger.info("Processing user audio transcript with memory agent")
                            print(f"\n[MEMORY INTEGRATION] Processing user audio transcript with memory agent")

                            # Check if this message might need memory context
                            # We're being more selective now - only check for memories when it seems relevant
                            retrieval_intent = self.memory_integration._detect_memory_retrieval_intent(transcript)

                            # Only check for location-related terms if the message is a question or seems to be asking about location
                            location_terms = ['where', 'live', 'location', 'address', 'city', 'state', 'country', 'from', 'reside']
                            contains_question = '?' in transcript or any(q in transcript.lower() for q in ['what', 'where', 'when', 'how', 'who', 'why'])
                            might_need_location = contains_question and any(term in transcript.lower() for term in location_terms)

                            # For location-related queries, we'll handle the response creation differently
                            # to prevent the "I don't know" followed by "Actually I do know" issue
                            if retrieval_intent or might_need_location:
                                if retrieval_intent:
                                    print(f"\n[MEMORY INTEGRATION] Detected memory retrieval intent: '{transcript[:50]}...'")
                                elif might_need_location:
                                    print(f"\n[MEMORY INTEGRATION] Detected location-related question: '{transcript[:50]}...'")

                                # For location queries, we'll skip the normal response creation
                                # and only create a response after memory retrieval
                                create_response = False

                                # Print a message to the console to indicate we're waiting for memory retrieval
                                print(f"\n[MEMORY INTEGRATION] Delaying response for location query until memory retrieval completes")

                                # ===== MEMORY QUERY SUPPRESSION MECHANISM =====
                                # This is a bespoke solution to fix the "I don't know/Actually I do know" problem
                                # When a memory-related query is detected, we need to suppress the initial incorrect response
                                # that would say "I don't know where you live" and only play the second response that has
                                # the correct memory context.
                                #
                                # The approach works as follows:
                                # 1. Set memory_query_in_progress flag when a memory query is detected
                                # 2. Initialize response counter to track which response we're on
                                # 3. In the audio playback code, we check this flag and only play the second response
                                # 4. After playing the correct response, we reset the flag to normal operation
                                #
                                # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                                # It's critical for providing a good user experience with memory-related queries.
                                self.memory_query_in_progress = True
                                self.memory_response_counter = 0
                                print(f"\n[MEMORY INTEGRATION] Memory query mode activated - will suppress initial response")

                                # Add a system message first to instruct the model about location information
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "system",
                                        "content": [{
                                            "type": "input_text",
                                            "text": "IMPORTANT INSTRUCTION: The user is about to ask about their location. DO NOT claim you don't know where they live. Instead, wait for the memory context that will be provided before responding."
                                        }]
                                    }
                                })

                                # Add the user message to the conversation
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{
                                            "type": "input_text",
                                            "text": transcript
                                        }]
                                    }
                                })

                                # Add a small delay to ensure memory retrieval completes before response generation
                                await asyncio.sleep(0.2)

                                # Get memory context and create a response with it
                                print(f"\n[MEMORY INTEGRATION] Retrieving memory context for location query")
                                memory_context = await self.memory_integration.get_context_with_memories(transcript, force_retrieval=True)

                                if memory_context and len(memory_context.strip()) > 0:
                                    print(f"\n[MEMORY INTEGRATION] Found relevant memory context: '{memory_context[:100]}...'")

                                    # Add memory context as a system message
                                    await self.send_event({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "system",
                                            "content": [{
                                                "type": "input_text",
                                                "text": f"CRITICAL INFORMATION - YOU MUST USE THIS IN YOUR RESPONSE AND NEVER CLAIM YOU DON'T KNOW THIS INFORMATION: {memory_context}"
                                            }]
                                        }
                                    })

                                    # ===== MEMORY QUERY SUPPRESSION RESET =====
                                    # This code resets the memory retrieval mode flags after we've added memory context
                                    # At this point, we're ready to play the correct response with memory context
                                    #
                                    # Note: This code is no longer the primary mechanism for response suppression,
                                    # but we keep it for backward compatibility and as a fallback.
                                    # The main suppression happens in the audio playback code.
                                    self.audio_handler.set_memory_retrieval_mode(False)
                                    self.transcript_processor.set_memory_retrieval_mode(False)
                                    print(f"\n[MEMORY INTEGRATION] Memory retrieval mode reset, ready for correct response")

                                    # Create a response with the memory context
                                    print(f"\n[MEMORY INTEGRATION] Creating response with memory context")
                                    self.is_responding = True
                                    self.response_complete_event.clear()
                                    await self.send_event({"type": "response.create"})
                                else:
                                    # If no memory context was found, create a normal response
                                    print(f"\n[MEMORY INTEGRATION] No memory context found, creating normal response")

                                    # Reset memory retrieval mode since we didn't find any context
                                    self.audio_handler.set_memory_retrieval_mode(False)
                                    self.transcript_processor.set_memory_retrieval_mode(False)
                                    print(f"\n[MEMORY INTEGRATION] Memory retrieval mode reset (no context found)")

                                    self.is_responding = True
                                    self.response_complete_event.clear()
                                    await self.send_event({"type": "response.create"})
                        except Exception as e:
                            logger.error(f"Error processing user audio transcript with memory agent: {e}")
                            print(f"\n[MEMORY INTEGRATION] Error processing user audio transcript: {e}")

                            # ===== MEMORY QUERY SUPPRESSION ERROR HANDLING =====
                            # This code ensures that if there's an error during memory query processing,
                            # we properly reset the memory query flags to prevent the system from getting stuck
                            # in a state where it's always suppressing responses.
                            #
                            # This is critical for error recovery and ensuring the system can continue
                            # to function normally after an error occurs.
                            if hasattr(self, 'memory_query_in_progress'):
                                self.memory_query_in_progress = False
                                self.memory_response_counter = 0
                                print(f"\n[MEMORY INTEGRATION] Memory query mode reset due to error")

                    # Create a response if needed (no memory was added)
                    if create_response:
                        # Send the conversation item create event
                        await self.send_event({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": transcript
                                }]
                            }
                        })

                        # Create the response
                        if not self.is_responding:
                            self.is_responding = True
                            self.response_complete_event.clear()
                            await self.send_event({"type": "response.create"})
                            print(f"\n[CONVERSATION] Creating response for: '{transcript[:50]}...'")

        # Handle errors and reset state if needed
        if event_type == "error":
            error_msg = event.get('error', {}).get('message', 'Unknown error')
            logger.error(f"Error event received: {error_msg}")
            print(f"\nError from API: {error_msg}")

            # Reset response state in case of errors
            if self.is_responding:
                self.reset_audio_state()

        # Handle response cancellation events
        elif event_type == "response.cancelled":
            # logger.info("Response was cancelled")
            print("\nResponse cancelled.")
            self.reset_audio_state()

        elif event_type == "response.text.delta":
            # Print text response incrementally
            if "delta" in event:
                delta_text = event["delta"]
                print(delta_text, end="", flush=True)

                # In chat mode, collect the response text
                if self.conversation_mode == "chat":
                    # Initialize response text if this is the first delta
                    if not hasattr(self, "current_response_text"):
                        self.current_response_text = ""
                    # Append the delta text
                    self.current_response_text += delta_text

                # In audio mode, also collect the response text for memory processing
                elif self.conversation_mode == "audio":
                    if not hasattr(self, "current_audio_response_text"):
                        self.current_audio_response_text = delta_text
                        print(f"\n[MEMORY INTEGRATION] Started collecting assistant audio response: '{delta_text}'")
                        logger.info(f"Started collecting assistant audio response text in audio mode")
                    else:
                        self.current_audio_response_text += delta_text
                        # Log periodically to avoid too much output
                        if len(self.current_audio_response_text) % 100 == 0:
                            logger.info(f"Collected {len(self.current_audio_response_text)} chars of assistant audio response")

        elif event_type == "response.audio.delta":
            # Set responding flag if we receive audio
            if not self.is_responding:
                self.is_responding = True
                self.response_complete_event.clear()

            # Append audio data to buffer
            audio_data = base64.b64decode(event.get("delta", ""))
            if audio_data:
                self.audio_buffer += audio_data

        # Handle user speech transcript events
        elif event_type == "input_audio_transcript.partial":
            # Handle partial transcript (doesn't need to be saved, just show progress)
            if "transcript" in event:
                transcript = event.get("transcript", "")
                if transcript:
                    print(f"\rRecognizing: {transcript}", end="", flush=True)

        elif event_type == "input_audio_transcript.final":
            # Handle final transcript of user speech
            if "transcript" in event:
                transcript = event.get("transcript", "")
                if transcript:
                    print(f"\nYou said: {transcript}")
                    # Add to transcript
                    self.transcript_processor.add_user_query(transcript)

                    # Flag to track if we need to create a response
                    create_response = True

                    # Process with memory agent if enabled
                    if self.enable_memory and self.memory_integration:
                        try:
                            # Process user message with memory agent
                            asyncio.create_task(self.memory_integration.process_user_message(transcript, self.conversation_mode))
                            logger.info("Processing user audio transcript with memory agent (final transcript)")
                            print(f"\n[MEMORY INTEGRATION] Processing user audio transcript with memory agent (final transcript)")

                            # Check if this message might need memory context
                            # We're being more selective now - only check for memories when it seems relevant
                            retrieval_intent = self.memory_integration._detect_memory_retrieval_intent(transcript)

                            # Only check for location-related terms if the message is a question or seems to be asking about location
                            location_terms = ['where', 'live', 'location', 'address', 'city', 'state', 'country', 'from', 'reside']
                            contains_question = '?' in transcript or any(q in transcript.lower() for q in ['what', 'where', 'when', 'how', 'who', 'why'])
                            might_need_location = contains_question and any(term in transcript.lower() for term in location_terms)

                            # For location-related queries, we'll handle the response creation differently
                            # to prevent the "I don't know" followed by "Actually I do know" issue
                            if retrieval_intent or might_need_location:
                                if retrieval_intent:
                                    print(f"\n[MEMORY INTEGRATION] Detected memory retrieval intent (final transcript): '{transcript[:50]}...'")
                                elif might_need_location:
                                    print(f"\n[MEMORY INTEGRATION] Detected location-related question (final transcript): '{transcript[:50]}...'")

                                # For location queries, we'll skip the normal response creation
                                # and only create a response after memory retrieval
                                create_response = False

                                # Print a message to the console to indicate we're waiting for memory retrieval
                                print(f"\n[MEMORY INTEGRATION] Delaying response for location query until memory retrieval completes")

                                # ===== MEMORY QUERY SUPPRESSION MECHANISM =====
                                # This is a bespoke solution to fix the "I don't know/Actually I do know" problem
                                # When a memory-related query is detected, we need to suppress the initial incorrect response
                                # that would say "I don't know where you live" and only play the second response that has
                                # the correct memory context.
                                #
                                # The approach works as follows:
                                # 1. Set memory_query_in_progress flag when a memory query is detected
                                # 2. Initialize response counter to track which response we're on
                                # 3. In the audio playback code, we check this flag and only play the second response
                                # 4. After playing the correct response, we reset the flag to normal operation
                                #
                                # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                                # It's critical for providing a good user experience with memory-related queries.
                                self.memory_query_in_progress = True
                                self.memory_response_counter = 0
                                print(f"\n[MEMORY INTEGRATION] Memory query mode activated - will suppress initial response")

                                # Add a system message first to instruct the model about location information
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "system",
                                        "content": [{
                                            "type": "input_text",
                                            "text": "IMPORTANT INSTRUCTION: The user is about to ask about their location. DO NOT claim you don't know where they live. Instead, wait for the memory context that will be provided before responding."
                                        }]
                                    }
                                })

                                # Add the user message to the conversation
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{
                                            "type": "input_text",
                                            "text": transcript
                                        }]
                                    }
                                })

                                # Add a small delay to ensure memory retrieval completes before response generation
                                await asyncio.sleep(0.2)

                                # Get memory context and create a response with it
                                print(f"\n[MEMORY INTEGRATION] Retrieving memory context for location query")
                                memory_context = await self.memory_integration.get_context_with_memories(transcript, force_retrieval=True)

                                if memory_context and len(memory_context.strip()) > 0:
                                    print(f"\n[MEMORY INTEGRATION] Found relevant memory context: '{memory_context[:100]}...'")

                                    # Add memory context as a system message
                                    await self.send_event({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "system",
                                            "content": [{
                                                "type": "input_text",
                                                "text": f"CRITICAL INFORMATION - YOU MUST USE THIS IN YOUR RESPONSE AND NEVER CLAIM YOU DON'T KNOW THIS INFORMATION: {memory_context}"
                                            }]
                                        }
                                    })

                                    # Create a response with the memory context
                                    print(f"\n[MEMORY INTEGRATION] Creating response with memory context")
                                    self.is_responding = True
                                    self.response_complete_event.clear()
                                    await self.send_event({"type": "response.create"})
                                else:
                                    # If no memory context was found, create a normal response
                                    print(f"\n[MEMORY INTEGRATION] No memory context found, creating normal response")
                                    self.is_responding = True
                                    self.response_complete_event.clear()
                                    await self.send_event({"type": "response.create"})
                        except Exception as e:
                            logger.error(f"Error processing user audio transcript with memory agent: {e}")
                            print(f"\n[MEMORY INTEGRATION] Error processing user audio transcript: {e}")

                    # Create a response if needed (no memory was added)
                    if create_response:
                        # Send the conversation item create event
                        await self.send_event({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": transcript
                                }]
                            }
                        })

                        # Create the response
                        if not self.is_responding:
                            self.is_responding = True
                            self.response_complete_event.clear()
                            await self.send_event({"type": "response.create"})
                            print(f"\n[CONVERSATION] Creating response for final transcript: '{transcript[:50]}...'")

        # Handle conversation item input audio transcription events
        elif event_type == "conversation.item.input_audio_transcription.completed":
            if "item" in event and "content" in event["item"]:
                for content in event["item"]["content"]:
                    if content.get("type") == "input_audio" and "transcript" in content:
                        transcript = content.get("transcript", "")
                        if transcript:
                            print(f"\nTranscribed: {transcript}")
                            # Add to transcript
                            self.transcript_processor.add_user_query(transcript)

                            # Flag to track if we need to create a response
                            create_response = True

                            # Process with memory agent if enabled
                            if self.enable_memory and self.memory_integration:
                                try:
                                    # Process user message with memory agent
                                    asyncio.create_task(self.memory_integration.process_user_message(transcript, self.conversation_mode))
                                    logger.info("Processing user audio transcript with memory agent (conversation item)")
                                    print(f"\n[MEMORY INTEGRATION] Processing user audio transcript with memory agent (conversation item)")

                                    # Check if this message might need memory context
                                    # We're being more selective now - only check for memories when it seems relevant
                                    retrieval_intent = self.memory_integration._detect_memory_retrieval_intent(transcript)

                                    # Only check for location-related terms if the message is a question or seems to be asking about location
                                    location_terms = ['where', 'live', 'location', 'address', 'city', 'state', 'country', 'from', 'reside']
                                    contains_question = '?' in transcript or any(q in transcript.lower() for q in ['what', 'where', 'when', 'how', 'who', 'why'])
                                    might_need_location = contains_question and any(term in transcript.lower() for term in location_terms)

                                    # For location-related queries, we'll handle the response creation differently
                                    # to prevent the "I don't know" followed by "Actually I do know" issue
                                    if retrieval_intent or might_need_location:
                                        if retrieval_intent:
                                            print(f"\n[MEMORY INTEGRATION] Detected memory retrieval intent (conversation item): '{transcript[:50]}...'")
                                        elif might_need_location:
                                            print(f"\n[MEMORY INTEGRATION] Detected location-related question (conversation item): '{transcript[:50]}...'")

                                        # For location queries, we'll skip the normal response creation
                                        # and only create a response after memory retrieval
                                        create_response = False

                                        # Print a message to the console to indicate we're waiting for memory retrieval
                                        print(f"\n[MEMORY INTEGRATION] Delaying response for location query until memory retrieval completes")

                                        # ===== MEMORY QUERY SUPPRESSION MECHANISM =====
                                        # This is a bespoke solution to fix the "I don't know/Actually I do know" problem
                                        # When a memory-related query is detected, we need to suppress the initial incorrect response
                                        # that would say "I don't know where you live" and only play the second response that has
                                        # the correct memory context.
                                        #
                                        # The approach works as follows:
                                        # 1. Set memory_query_in_progress flag when a memory query is detected
                                        # 2. Initialize response counter to track which response we're on
                                        # 3. In the audio playback code, we check this flag and only play the second response
                                        # 4. After playing the correct response, we reset the flag to normal operation
                                        #
                                        # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                                        # It's critical for providing a good user experience with memory-related queries.
                                        self.memory_query_in_progress = True
                                        self.memory_response_counter = 0
                                        print(f"\n[MEMORY INTEGRATION] Memory query mode activated - will suppress initial response")

                                        # Add a system message first to instruct the model about location information
                                        await self.send_event({
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "message",
                                                "role": "system",
                                                "content": [{
                                                    "type": "input_text",
                                                    "text": "IMPORTANT INSTRUCTION: The user is about to ask about their location. DO NOT claim you don't know where they live. Instead, wait for the memory context that will be provided before responding."
                                                }]
                                            }
                                        })

                                        # Add the user message to the conversation
                                        await self.send_event({
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "message",
                                                "role": "user",
                                                "content": [{
                                                    "type": "input_text",
                                                    "text": transcript
                                                }]
                                            }
                                        })

                                        # Add a small delay to ensure memory retrieval completes before response generation
                                        await asyncio.sleep(0.2)

                                        # Get memory context and create a response with it
                                        print(f"\n[MEMORY INTEGRATION] Retrieving memory context for location query")
                                        memory_context = await self.memory_integration.get_context_with_memories(transcript, force_retrieval=True)

                                        if memory_context and len(memory_context.strip()) > 0:
                                            print(f"\n[MEMORY INTEGRATION] Found relevant memory context: '{memory_context[:100]}...'")

                                            # Add memory context as a system message
                                            await self.send_event({
                                                "type": "conversation.item.create",
                                                "item": {
                                                    "type": "message",
                                                    "role": "system",
                                                    "content": [{
                                                        "type": "input_text",
                                                        "text": f"CRITICAL INFORMATION - YOU MUST USE THIS IN YOUR RESPONSE AND NEVER CLAIM YOU DON'T KNOW THIS INFORMATION: {memory_context}"
                                                    }]
                                                }
                                            })

                                            # ===== MEMORY QUERY SUPPRESSION RESET =====
                                            # This code resets the memory retrieval mode flags after we've added memory context
                                            # At this point, we're ready to play the correct response with memory context
                                            #
                                            # Note: This code is no longer the primary mechanism for response suppression,
                                            # but we keep it for backward compatibility and as a fallback.
                                            # The main suppression happens in the audio playback code.
                                            self.audio_handler.set_memory_retrieval_mode(False)
                                            self.transcript_processor.set_memory_retrieval_mode(False)
                                            print(f"\n[MEMORY INTEGRATION] Memory retrieval mode reset, ready for correct response")

                                            # Create a response with the memory context
                                            print(f"\n[MEMORY INTEGRATION] Creating response with memory context")
                                            self.is_responding = True
                                            self.response_complete_event.clear()
                                            await self.send_event({"type": "response.create"})
                                        else:
                                            # If no memory context was found, create a normal response
                                            print(f"\n[MEMORY INTEGRATION] No memory context found, creating normal response")

                                            # Reset memory retrieval mode since we didn't find any context
                                            self.audio_handler.set_memory_retrieval_mode(False)
                                            self.transcript_processor.set_memory_retrieval_mode(False)
                                            print(f"\n[MEMORY INTEGRATION] Memory retrieval mode reset (no context found)")

                                            self.is_responding = True
                                            self.response_complete_event.clear()
                                            await self.send_event({"type": "response.create"})
                                except Exception as e:
                                    logger.error(f"Error processing user audio transcript with memory agent: {e}")
                                    print(f"\n[MEMORY INTEGRATION] Error processing user audio transcript: {e}")

                                    # ===== MEMORY QUERY SUPPRESSION ERROR HANDLING =====
                                    # This code ensures that if there's an error during memory query processing,
                                    # we properly reset the memory query flags to prevent the system from getting stuck
                                    # in a state where it's always suppressing responses.
                                    #
                                    # This is critical for error recovery and ensuring the system can continue
                                    # to function normally after an error occurs.
                                    if hasattr(self, 'memory_query_in_progress'):
                                        self.memory_query_in_progress = False
                                        self.memory_response_counter = 0
                                        print(f"\n[MEMORY INTEGRATION] Memory query mode reset due to error")

                            # Create a response if needed (no memory was added)
                            if create_response:
                                # Send the conversation item create event
                                await self.send_event({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{
                                            "type": "input_text",
                                            "text": transcript
                                        }]
                                    }
                                })

                                # Create the response
                                if not self.is_responding:
                                    self.is_responding = True
                                    self.response_complete_event.clear()
                                    await self.send_event({"type": "response.create"})
                                    print(f"\n[CONVERSATION] Creating response for conversation item: '{transcript[:50]}...'")

        elif event_type == "response.audio.done":
            # Play the complete audio response
            if self.audio_buffer:
                # logger.info(f"Audio response complete, playing {len(self.audio_buffer)} bytes")

                # --- Send model audio chunk to sentiment analysis ---
                if self.enable_sentiment_analysis and self.audio_mp_queue:
                    model_audio_chunk = {
                        "source": "model",
                        "data": self.audio_buffer,
                        "rate": self.audio_handler.rate, # Assuming output rate matches input rate here
                        "timestamp": time.time()
                    }
                    try:
                        self.audio_mp_queue.put_nowait(model_audio_chunk)
                    except queue.Full:
                        logger.warning("Sentiment analysis audio queue is full. Skipping model chunk.")
                    except Exception as e:
                        logger.error(f"Error sending model chunk to sentiment queue: {e}")
                # ---------------------------------------------------

                # ===== MEMORY QUERY RESPONSE SUPPRESSION IMPLEMENTATION =====
                # This is the core of our solution to the "I don't know/Actually I do know" problem
                # When processing a memory-related query, the model first generates an incorrect response
                # saying it doesn't know, then after receiving memory context, it generates a correct response.
                #
                # This code ensures that:
                # 1. We track which response we're on using memory_response_counter
                # 2. We SKIP the first response completely (the incorrect one)
                # 3. We ONLY play the second response (the correct one with memory context)
                # 4. We reset the flags after playing the correct response
                #
                # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                # It's critical for providing a good user experience with memory-related queries.
                if hasattr(self, 'memory_query_in_progress') and self.memory_query_in_progress:
                    # Increment the response counter
                    if not hasattr(self, 'memory_response_counter'):
                        self.memory_response_counter = 0
                    self.memory_response_counter += 1

                    # Only play the second response (the corrected one with memory context)
                    if self.memory_response_counter >= 2:
                        print(f"\n[MEMORY INTEGRATION] Playing corrected response with memory context")
                        self.audio_handler.play_audio(self.audio_buffer)
                        # Reset the memory query flag after playing the corrected response
                        self.memory_query_in_progress = False
                        self.memory_response_counter = 0
                    else:
                        print(f"\n[MEMORY INTEGRATION] Suppressing initial incorrect response")
                        # Don't play the first response - this is intentional!
                        # The first response would say "I don't know" which is incorrect
                else:
                    # Normal response, play it
                    self.audio_handler.play_audio(self.audio_buffer)

                self.audio_buffer = b'' # Clear buffer AFTER sending and playing

            # Final check to play any buffered audio that wasn't caught by response.audio.done
            if self.audio_buffer:
                # logger.info(f"Playing remaining buffered audio ({len(self.audio_buffer)} bytes)")

                # --- Send remaining model audio chunk to sentiment analysis ---
                if self.enable_sentiment_analysis and self.audio_mp_queue:
                    model_audio_chunk = {
                        "source": "model",
                        "data": self.audio_buffer,
                        "rate": self.audio_handler.rate, # Assuming output rate matches input rate
                        "timestamp": time.time()
                    }
                    try:
                        self.audio_mp_queue.put_nowait(model_audio_chunk)
                    except queue.Full:
                        logger.warning("Sentiment analysis audio queue is full. Skipping final model chunk.")
                    except Exception as e:
                        logger.error(f"Error sending final model chunk to sentiment queue: {e}")
                # -------------------------------------------------------------

                # ===== MEMORY QUERY RESPONSE SUPPRESSION IMPLEMENTATION (SECOND INSTANCE) =====
                # This is a duplicate of the code above, but for the remaining audio buffer
                # It's important to maintain both instances to ensure complete suppression
                # of incorrect responses in all audio playback paths.
                #
                # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                # It's critical for providing a good user experience with memory-related queries.
                if hasattr(self, 'memory_query_in_progress') and self.memory_query_in_progress:
                    # Increment the response counter
                    if not hasattr(self, 'memory_response_counter'):
                        self.memory_response_counter = 0
                    self.memory_response_counter += 1

                    # Only play the second response (the corrected one with memory context)
                    if self.memory_response_counter >= 2:
                        print(f"\n[MEMORY INTEGRATION] Playing corrected response with memory context")
                        self.audio_handler.play_audio(self.audio_buffer)
                        # Reset the memory query flag after playing the corrected response
                        self.memory_query_in_progress = False
                        self.memory_response_counter = 0
                    else:
                        print(f"\n[MEMORY INTEGRATION] Suppressing initial incorrect response")
                        # Don't play the first response - this is intentional!
                        # The first response would say "I don't know" which is incorrect
                else:
                    # Normal response, play it
                    self.audio_handler.play_audio(self.audio_buffer)

                self.audio_buffer = b'' # Clear buffer AFTER sending and playing

            # Handle chat mode response completion
            if self.conversation_mode == "chat" and hasattr(self, "current_response_text"):
                # Add the complete response to chat history
                self.chat_history.append({"role": "assistant", "content": self.current_response_text})

                # Process the response text for sentiment analysis
                if self.enable_sentiment_analysis:
                    asyncio.create_task(self._process_text_for_sentiment(self.current_response_text, "model"))

                # Add to transcript
                self.transcript_processor.add_assistant_response(self.current_response_text)

                # Reset the current response text
                self.current_response_text = ""

            # Handle audio mode response completion
            elif self.conversation_mode == "audio" and hasattr(self, "current_audio_response_text"):
                # Process the complete audio response text with memory agent
                if self.enable_memory and self.memory_integration and self.current_audio_response_text:
                    try:
                        # Add to transcript first
                        self.transcript_processor.add_assistant_response(self.current_audio_response_text)

                        print(f"\n[MEMORY INTEGRATION] Processing assistant audio response with memory agent: '{self.current_audio_response_text[:100]}...'")

                        # Process with memory agent
                        asyncio.create_task(self.memory_integration.process_assistant_message(
                            self.current_audio_response_text,
                            self.conversation_mode
                        ))
                        logger.info("Processing audio response with memory agent")
                    except Exception as e:
                        logger.error(f"Error processing audio response with memory agent: {e}")
                        print(f"\n[MEMORY INTEGRATION] Error processing assistant audio response: {e}")
                else:
                    # Log why we're not processing
                    if not self.enable_memory:
                        print("\n[MEMORY INTEGRATION] Memory system is disabled for audio mode")
                    elif not self.memory_integration:
                        print("\n[MEMORY INTEGRATION] Memory integration is not initialized for audio mode")
                    elif not self.current_audio_response_text:
                        print("\n[MEMORY INTEGRATION] No assistant audio response text to process")
                    logger.warning(f"Not processing assistant audio response: enable_memory={self.enable_memory}, memory_integration={self.memory_integration is not None}, response_text_length={len(self.current_audio_response_text) if hasattr(self, 'current_audio_response_text') else 0}")

                # Reset the current audio response text
                self.current_audio_response_text = ""

            print("Updating transcript...")
            self.transcript_processor.update_transcript()
            print("Transcript updated in memory.")

            # --- Retrieve and print sentiment results ---
            if self.enable_sentiment_analysis and self.sentiment_history_list is not None:
                 # Convert shared list to regular list for printing
                 final_sentiment_results = list(self.sentiment_history_list)
                 if final_sentiment_results:
                      print("\n--- Sentiment Analysis Results ---")
                      for result in final_sentiment_results:
                          if 'error' in result:
                               ts_err = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                               print(f"[{ts_err}] Sentiment Analysis ERROR: {result['error']}")
                               if 'traceback' in result:
                                    print(result['traceback'])
                          else:
                              ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                              duration = result.get('audio_duration_s', 'N/A')
                              label = result.get('sentiment_label', 'N/A')
                              value = result.get('sentiment_value', 'N/A')
                              raw = result.get('raw_response', '')
                              # Format duration safely
                              duration_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else str(duration)
                              print(f"[{ts}] Label: {label} ({value}), Duration: {duration_str}, Raw: '{raw}'")
                      print("----------------------------------")
                 else:
                      print("\nNo sentiment analysis results were generated.")
            # -------------------------------------------

            # Signal that the response is complete
            self.is_responding = False
            self.response_complete_event.set()
            print("\nResponse complete. You can speak again or type a command.")

            # Check if we're waiting to add memory context
            if hasattr(self, 'waiting_to_add_memory') and self.waiting_to_add_memory and hasattr(self, 'memory_context_to_add'):
                print(f"\n[MEMORY INTEGRATION] Adding delayed memory context as follow-up")

                # Reset the flag
                self.waiting_to_add_memory = False

                # Add the memory context as a system message
                memory_context = self.memory_context_to_add

                # Create a task to add the memory context and generate a follow-up response
                asyncio.create_task(self._add_followup_memory_response(memory_context))

        # Handle WebRTC/Realtime API voice activity events
        elif event_type == "input_audio_buffer.speech_started":
            # logger.info("Speech started detected by server VAD")
            print("\nSpeech detected ...")
            # If we're currently responding and speech is detected, reset audio state
            if self.is_responding and self.VAD_config.get("interrupt_response", False):
                self.reset_audio_state()

            # Add to transcript
            self.transcript_processor.add_speech_event("speech_started")

        elif event_type == "input_audio_buffer.speech_stopped":
            # logger.info("Speech stopped detected by server VAD")
            print("\nSpeech ended, processing ...")
            # Add to transcript
            self.transcript_processor.add_speech_event("speech_stopped")

        # Handle session events which can contain important connection info
        elif event_type in ["session.created", "session.updated"]:
            # logger.info(f"Session {event_type.split('.')[1]}: {event.get('session', {}).get('id', 'unknown')}")
            # Check if turn detection settings are present
            if 'session' in event and 'turn_detection' in event['session']:
                turn_detection = event['session']['turn_detection']
                # logger.info(f"Turn detection settings: {turn_detection}")

        # Route transcript events to the separate processor
        elif event_type.startswith("response.audio_transcript"):
            # Send to transcript processor thread instead of handling here
            self.transcript_processor.add_transcript_event(event)

        # Handle rate limits updates
        elif event_type == "rate_limits.updated":
            rate_limits = event.get("rate_limits", [])
            for limit in rate_limits:
                logger.info(f"Rate limit: {limit.get('type')} - {limit.get('bucket')}: {limit.get('remaining')} remaining, resets in {limit.get('reset_seconds')} seconds")

        # Check if we need to keep recording or stop
        if event_type in ["input_audio_buffer.speech_stopped", "response.done"]:
            # These events indicate the server has detected the end of speech
            # or the model has finished responding
            # We can add specific actions here if needed
            pass

    async def send_text(self, text):
        """Send a text message to the appropriate API based on conversation mode."""
        # logger.info(f"Sending text message: {text}")

        # If in chat mode, use the chat completions API
        if self.conversation_mode == "chat":
            await self.send_chat_message(text)
            return

        # For audio mode, continue with the realtime API
        # Wait for any current response to complete
        if self.is_responding:
            print("Assistant is still responding. Do you want to interrupt? (y/n)")
            try:
                choice = await asyncio.to_thread(input)
                if choice.lower() != 'y':
                    print("Waiting for current response to complete...")
                    await self.response_complete_event.wait()
                else:
                    print("Interrupting current response...")
                    # Cancel the current response before sending a new one
                    await self.send_event({"type": "response.cancel"})
                    self.reset_audio_state()
                    await asyncio.sleep(0.5)  # Brief pause to allow cancellation to process
            except Exception as e:
                logger.error(f"Error getting interruption choice: {e}")
                # Default to waiting if we can't get input
                await self.response_complete_event.wait()

        # Add the text query to the transcript
        self.transcript_processor.add_user_query(text)

        # Process message with memory agent if enabled
        if self.enable_memory and self.memory_integration:
            try:
                await self.memory_integration.process_user_message(text, self.conversation_mode)
                logger.info("Processed user message with memory agent")
            except Exception as e:
                logger.error(f"Error processing message with memory agent: {e}")

        # Prepare the message content
        message_content = [{
            "type": "input_text",
            "text": text
        }]

        # Check if the message contains phrases indicating memory retrieval intent
        retrieval_intent = False
        if self.enable_memory and self.memory_integration:
            try:
                retrieval_intent = self.memory_integration._detect_memory_retrieval_intent(text)
                if retrieval_intent:
                    logger.info("Detected memory retrieval intent in user message")
            except Exception as e:
                logger.error(f"Error detecting memory retrieval intent: {e}")

        # Add memory context if enabled
        if self.enable_memory and self.memory_integration:
            try:
                print(f"\n[MEMORY INTEGRATION] Retrieving memory context for: '{text[:50]}...'")
                logger.info(f"Retrieving memory context in {self.conversation_mode} mode")
                memory_context = await self.memory_integration.get_context_with_memories(text, retrieval_intent)
                if memory_context:
                    print(f"\n[MEMORY INTEGRATION] Adding memory context to conversation: '{memory_context[:100]}...'")
                    # Add memory context as a system message
                    await self.send_event({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "system",
                            "content": [{
                                "type": "input_text",
                                "text": memory_context
                            }]
                        }
                    })
                    logger.info("Added memory context to conversation")
                else:
                    print(f"\n[MEMORY INTEGRATION] No relevant memory context found for: '{text[:50]}...'")
                    logger.info("No relevant memory context found")
            except Exception as e:
                logger.error(f"Error adding memory context: {e}")
                print(f"\n[MEMORY INTEGRATION] Error retrieving memory context: {e}")

        # Send the user message
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": message_content
            }
        })

        # Make sure we're not in responding state
        if not self.is_responding:
            # Set responding state before creating response
            self.is_responding = True
            self.response_complete_event.clear()

            await self.send_event({"type": "response.create"})
        else:
            logger.warning("Not sending response.create as another response is in progress")

    async def _add_followup_memory_response(self, memory_context):
        """Add a follow-up response with memory context after the initial response completes.

        Args:
            memory_context (str): The memory context to add to the follow-up response
        """
        try:
            # Add a small delay to ensure the previous response is fully processed
            await asyncio.sleep(0.5)

            # Add memory context as a system message with high priority
            print(f"\n[MEMORY INTEGRATION] Sending memory context as system message for follow-up")
            await self.send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{
                        "type": "input_text",
                        "text": f"CRITICAL INFORMATION - YOU MUST USE THIS IN YOUR RESPONSE AND NEVER CLAIM YOU DON'T KNOW THIS INFORMATION: {memory_context}"
                    }]
                }
            })
            logger.info("Added memory context for follow-up response")

            # Add a delay to ensure the memory context is processed
            await asyncio.sleep(0.5)

            # Create a new response that will use the memory context
            print(f"\n[MEMORY INTEGRATION] Creating follow-up response with memory context")
            self.is_responding = True
            self.response_complete_event.clear()

            # Create a special follow-up message that sounds natural
            # This is a more generic follow-up that doesn't assume the context is about Seattle
            synthetic_message = "Is there anything else you'd like to tell me?"

            # Add the synthetic message to the transcript first
            print(f"\n[MEMORY INTEGRATION] Adding synthetic follow-up message to transcript: '{synthetic_message}'")
            self.transcript_processor.add_user_query(f"[System-generated follow-up]: {synthetic_message}")

            # Then send it to the API
            await self.send_event({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": synthetic_message
                    }]
                }
            })

            # Create the response
            await self.send_event({"type": "response.create"})

            return True
        except Exception as e:
            logger.error(f"Error adding follow-up memory response: {e}")
            print(f"\n[MEMORY INTEGRATION] Error adding follow-up memory response: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return False

    async def _add_memory_context_for_audio(self, text, force_retrieval=False):
        """Retrieve and add memory context for audio transcripts.

        Args:
            text (str): The user's audio transcript
            force_retrieval (bool): Whether to force memory retrieval even if no intent is detected
        """
        if not self.enable_memory or not self.memory_integration:
            return False

        try:
            print(f"\n[MEMORY INTEGRATION] Retrieving memory context for audio transcript: '{text[:50]}...'")
            logger.info(f"Retrieving memory context for audio transcript in {self.conversation_mode} mode")

            # Only force retrieval when explicitly requested
            # This makes memory retrieval more selective and contextual
            memory_context = await self.memory_integration.get_context_with_memories(text, force_retrieval=force_retrieval)

            if memory_context and len(memory_context.strip()) > 0:
                print(f"\n[MEMORY INTEGRATION] Found relevant memory context: '{memory_context[:100]}...'")
                logger.info("Found relevant memory context for audio transcript")

                # Check if we're currently responding
                if self.is_responding:
                    # Instead of cancelling, wait for the current response to complete
                    print(f"\n[MEMORY INTEGRATION] Waiting for current response to complete before adding memory context")

                    # Set a flag to indicate we're waiting to add memory context
                    self.waiting_to_add_memory = True
                    self.memory_context_to_add = memory_context

                    # Return true to indicate memory context will be added (but after current response)
                    return True

                # If we're not currently responding, add the memory context now
                # Add memory context as a system message with high priority
                print(f"\n[MEMORY INTEGRATION] Sending memory context as system message")
                await self.send_event({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "system",
                        "content": [{
                            "type": "input_text",
                            "text": f"CRITICAL INFORMATION - YOU MUST USE THIS IN YOUR RESPONSE AND NEVER CLAIM YOU DON'T KNOW THIS INFORMATION: {memory_context}"
                        }]
                    }
                })
                logger.info("Added memory context to audio conversation")

                # Add a delay to ensure the memory context is processed
                await asyncio.sleep(0.5)

                # Create a new response to ensure the memory context is used
                print(f"\n[MEMORY INTEGRATION] Creating new response with memory context")
                self.is_responding = True
                self.response_complete_event.clear()
                await self.send_event({"type": "response.create"})

                # Return true to indicate memory context was added
                return True
            else:
                print(f"\n[MEMORY INTEGRATION] No relevant memory context found for audio transcript: '{text[:50]}...'")
                logger.info("No relevant memory context found for audio transcript")
                return False
        except Exception as e:
            logger.error(f"Error adding memory context for audio transcript: {e}")
            print(f"\n[MEMORY INTEGRATION] Error retrieving memory context for audio transcript: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return False

    async def _process_text_for_sentiment(self, text, source):
        """Process text for sentiment analysis.

        Args:
            text (str): The text to analyze
            source (str): Either 'user' or 'model'
        """
        if not self.enable_sentiment_analysis or not self.audio_mp_queue:
            return

        try:
            # Create a text-based sentiment analysis request
            text_sentiment_request = {
                "source": source,
                "text": text,
                "timestamp": time.time()
            }

            # Send to the sentiment analysis queue
            self.audio_mp_queue.put_nowait(text_sentiment_request)
            logger.info(f"Sent {source} text for sentiment analysis: {text[:30]}...")
        except queue.Full:
            logger.warning("Sentiment analysis queue is full. Skipping text analysis.")
        except Exception as e:
            logger.error(f"Error sending text for sentiment analysis: {e}")

    async def send_chat_message(self, text):
        """Send a message using the chat completions API instead of the realtime API.

        Args:
            text (str): The text message to send
        """
        # Add the text query to the transcript
        self.transcript_processor.add_user_query(text)

        # Add to chat history
        self.chat_history.append({"role": "user", "content": text})

        # Send text for sentiment analysis if enabled
        if self.enable_sentiment_analysis:
            await self._process_text_for_sentiment(text, "user")

        # Process message with memory agent if enabled
        if self.enable_memory and self.memory_integration:
            try:
                await self.memory_integration.process_user_message(text, self.conversation_mode)
                logger.info("Processed user message with memory agent")
            except Exception as e:
                logger.error(f"Error processing message with memory agent: {e}")

        # Set responding state
        self.is_responding = True
        self.response_complete_event.clear()

        try:
            # Print user message
            print(f"\nYou: {text}")

            # Create messages for the API call
            messages = []

            # Add system message with instructions about memory capabilities
            # Create personalized instructions if username is available
            user_greeting = f"The user's name is {self.username}." if self.username else ""

            system_message = {
                "role": "system",
                "content": f"You are a helpful AI assistant with access to memories from previous conversations. {user_greeting} Use these memories to provide personalized and contextually relevant responses. When appropriate, reference these past interactions naturally.\n\nYou have the ability to create new memories about the user. When you want to explicitly remember something important about the user, you can use phrases like 'I'll remember that about you' or 'I'll make a note of that.' When you use these phrases, the system will automatically store this information in your memory. This is particularly useful for important user preferences, personal details, or significant events that would be helpful to recall in future conversations.\n\nYou can also explicitly trigger memory retrieval by using phrases like 'let me recall' or 'let me see what I know about you' or 'based on our previous conversations'. When you use these phrases, the system will prioritize retrieving relevant memories to help you provide more personalized responses.\n\nIf you attempt to recall information but no relevant memories are found, you will receive a system message starting with '### Memory retrieval note:'. In such cases, acknowledge that you don't have that specific information in your memory and respond based on your general knowledge instead. Be honest about what you don't know about the user's past interactions."
            }
            messages.append(system_message)

            # Add chat history
            for msg in self.chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Check if the message contains phrases indicating memory retrieval intent
            retrieval_intent = False
            if self.enable_memory and self.memory_integration:
                try:
                    retrieval_intent = self.memory_integration._detect_memory_retrieval_intent(text)
                    if retrieval_intent:
                        logger.info("Detected memory retrieval intent in chat message")
                except Exception as e:
                    logger.error(f"Error detecting memory retrieval intent: {e}")

            # Enhance messages with memories if enabled
            if self.enable_memory and self.memory_integration:
                try:
                    enhanced_messages = await self.memory_integration.enhance_chat_messages(messages, text)
                    messages = enhanced_messages
                    logger.info("Enhanced chat messages with memories")
                except Exception as e:
                    logger.error(f"Error enhancing messages with memories: {e}")

            # Call the chat completions API
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.7
            )

            # Get the response text
            response_text = response.choices[0].message.content

            # Print the response
            print(f"\nAssistant: {response_text}")

            # Add to chat history
            self.chat_history.append({"role": "assistant", "content": response_text})

            # Add to transcript
            self.transcript_processor.add_assistant_response(response_text)

            # Process for sentiment analysis
            if self.enable_sentiment_analysis:
                await self._process_text_for_sentiment(response_text, "model")

            # Process assistant message with memory agent if enabled
            if self.enable_memory and self.memory_integration:
                try:
                    await self.memory_integration.process_assistant_message(response_text, self.conversation_mode)
                    logger.info("Processed assistant message with memory agent")
                except Exception as e:
                    logger.error(f"Error processing assistant message with memory agent: {e}")

            # Update transcript in memory
            print("\nUpdating transcript...")
            self.transcript_processor.update_transcript()
            print("Transcript updated in memory.")

            # Print sentiment results if available
            if self.enable_sentiment_analysis and hasattr(self, 'sentiment_process') and self.sentiment_process is not None:
                print("\nSentiment analysis is enabled. Results will be shown in the emotion radar chart.")

            # Print memory status if enabled
            if self.enable_memory and self.memory_integration:
                print("\nMemory system is active. Relevant memories will be used in future conversations.")

        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            print(f"\nError: {str(e)}")
            traceback.print_exc()
        finally:
            # Reset responding state
            self.is_responding = False
            self.response_complete_event.set()
            print("\nResponse complete. You can type another message or use a command.")

    async def send_audio(self):
        """Record and send audio using server-side turn detection."""
        # logger.info("Starting continuous listening mode...")

        # Wait for any current response to complete - with a note about interruption
        if self.is_responding:
            print("Assistant is responding. You can continue to interrupt or wait...")
            print("Press Enter to interrupt the current response.")
            # We'll allow interruption through the wait_for_input function

        # If we're still in responding state but want to interrupt, cancel it
        if self.is_responding:
            try:
                await self.send_event({"type": "response.cancel"})
                self.reset_audio_state()
                await asyncio.sleep(0.5)  # Brief pause
            except Exception as e:
                logger.error(f"Error cancelling response: {e}")

        print("You can now speak. Press Enter to stop listening mode or 'q' to quit.")

        # Clear transcript from previous sessions
        self.transcript_processor.clear_transcript()

        # Reset the transcript received flag for this audio session
        self.received_transcript = False

        self.audio_handler.start_recording()

        # For stopping conditions
        stop_recording = False
        quit_requested = False
        speech_detected = False

        # Track if we received any transcripts
        received_transcript = [False]

        # For transcript display
        transcript_display_task = None

        async def wait_for_input():
            nonlocal stop_recording, quit_requested
            user_input = await asyncio.to_thread(input)
            if user_input.lower() == 'q':
                print("\n=== Quit requested - EMERGENCY STOP ===")
                quit_requested = True

                # Set the shutdown flag to prevent any further event processing
                self.shutdown_in_progress = True

                # EMERGENCY STOP: Forcefully terminate all audio processes immediately
                print("EMERGENCY STOP: Forcefully terminating all audio processes...")
                self.audio_handler.emergency_stop()

                # Save the transcript immediately
                print("Saving transcript...")
                transcript_file = self.transcript_processor.save_transcript(force=True)
                if transcript_file:
                    print(f"Transcript saved to: {transcript_file}")

                # Immediately cancel any ongoing response
                if self.is_responding:
                    try:
                        # Force reset the response state
                        self.is_responding = False
                        self.response_complete_event.set()

                        # Try to cancel the response via WebSocket
                        if self.ws:
                            try:
                                await self.send_event({"type": "response.cancel"})
                            except Exception:
                                # Ignore errors during shutdown
                                pass
                    except Exception as e:
                        logger.error(f"Error cancelling response during quit: {e}")
            stop_recording = True

        async def display_transcript_updates():
            """Periodically check for transcript updates (no longer displaying to console)."""
            last_transcript = ""
            while not stop_recording and self.ws:
                # Still check for updates but don't print to console
                current_transcript = self.transcript_processor.get_current_transcript()
                if current_transcript and current_transcript != last_transcript:
                    # Just update the last transcript without printing
                    last_transcript = current_transcript
                await asyncio.sleep(0.5)  # Check every half second

        try:
            # Start the input listener in the background
            enter_task = asyncio.create_task(wait_for_input())

            # Start transcript display task
            transcript_display_task = asyncio.create_task(display_transcript_updates())

            # Keep track of how long we've been recording
            start_time = time.time()
            last_chunk_time = start_time
            last_activity_time = start_time
            silent_chunks = 0

            # Buffer for audio data if no transcript is received
            recorded_audio_data = b''

            while not stop_recording and self.ws:
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    try:
                        # Save a copy of the audio data for fallback
                        recorded_audio_data += chunk

                        # --- Send user audio chunk to sentiment analysis ---
                        if self.enable_sentiment_analysis and self.audio_mp_queue:
                            user_audio_chunk = {
                                "source": "user",
                                "data": chunk,
                                "rate": self.audio_handler.rate, # Get rate from handler
                                "timestamp": time.time()
                            }
                            try:
                                self.audio_mp_queue.put_nowait(user_audio_chunk)
                            except queue.Full:
                                logger.warning("Sentiment analysis audio queue is full. Skipping chunk.")
                            except Exception as e:
                                logger.error(f"Error sending chunk to sentiment queue: {e}")
                        # -------------------------------------------------

                        # Check if the chunk has actual audio or just silence
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        max_amp = np.max(np.abs(audio_array))

                        # Log audio levels less frequently to reduce console spam
                        current_time = time.time()
                        if current_time - last_chunk_time > 2.0:  # Log every 2 seconds
                            last_chunk_time = current_time

                        # Count silent chunks - useful for debugging
                        if max_amp < 100:
                            silent_chunks += 1
                        else:
                            silent_chunks = 0
                            last_activity_time = current_time

                        # Encode and send to the server as we go
                        base64_chunk = base64.b64encode(chunk).decode('utf-8')
                        await self.send_event({
                            "type": "input_audio_buffer.append",
                            "audio": base64_chunk
                        })

                        # --- Send user audio chunk to sentiment analysis ---
                        # Also send chunk here for manual mode
                        if self.enable_sentiment_analysis and self.audio_mp_queue:
                            user_audio_chunk = {
                                "source": "user",
                                "data": chunk,
                                "rate": self.audio_handler.rate, # Get rate from handler
                                "timestamp": time.time()
                            }
                            try:
                                self.audio_mp_queue.put_nowait(user_audio_chunk)
                            except queue.Full:
                                logger.warning("Sentiment analysis audio queue is full (manual mode). Skipping chunk.")
                            except Exception as e:
                                logger.error(f"Error sending chunk to sentiment queue (manual mode): {e}")
                        # -------------------------------------------------

                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error sending audio chunk: {e}")
                        if "WebSocket connection is closed" in str(e):
                            logger.warning("WebSocket connection closed, reconnecting...")
                            try:
                                # Try to reconnect
                                connection_success = await self.connect()
                                if not connection_success:
                                    logger.error("Failed to reconnect. Stopping audio recording.")
                                    break
                                # logger.info("Successfully reconnected!")
                                continue
                            except Exception as reconnect_error:
                                logger.error(f"Error during reconnection: {reconnect_error}")
                                break
                        else:
                            break  # Exit loop on other errors
                else:
                    await asyncio.sleep(0.1)

                # Safety timeout - if recording for too long without stopping
                if current_time - start_time > 300.0:  # 5 minute maximum
                    logger.warning("Recording timeout reached (5 minutes), stopping automatically")
                    stop_recording = True

                # Send periodic ping messages if no activity for more than 30 seconds
                if current_time - last_activity_time > 30.0:
                    try:
                        # Send a small dummy audio buffer as a keepalive
                        dummy_silent_chunk = np.zeros(1024, dtype=np.int16).tobytes()
                        base64_chunk = base64.b64encode(dummy_silent_chunk).decode('utf-8')
                        await self.send_event({
                            "type": "input_audio_buffer.append",
                            "audio": base64_chunk
                        })
                        last_activity_time = current_time
                    except Exception as e:
                        logger.error(f"Error sending keepalive: {e}")

            # Cancel enter_task if it's still running
            if not enter_task.done():
                enter_task.cancel()
                try:
                    await enter_task
                except asyncio.CancelledError:
                    pass

            # Cancel transcript display task
            if transcript_display_task and not transcript_display_task.done():
                transcript_display_task.cancel()
                try:
                    await transcript_display_task
                except asyncio.CancelledError:
                    pass

            # logger.info(f"Audio recording finished after {time.time() - start_time:.1f} seconds")

            # If we didn't receive any transcripts but have recorded audio,
            # add a placeholder in the transcript
            if not self.received_transcript and len(recorded_audio_data) > 1024:  # Make sure we have some meaningful audio
                # Add a fallback entry indicating speech was recorded but not transcribed
                self.transcript_processor.add_user_query("[Speech recorded but not transcribed]")
                logger.warning("No transcript received from API, added placeholder")

        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Stop recording
            self.audio_handler.stop_recording()

        # If using manual VAD, we need to commit
        if not self.VAD_turn_detection and self.ws:
            try:
                await self.send_event({"type": "input_audio_buffer.commit"})
            except Exception as e:
                logger.error(f"Error committing audio buffer: {e}")

        # Always explicitly create a response if server-VAD didn't already do it
        try:
            if self.ws and not self.is_responding:
                self.is_responding = True
                self.response_complete_event.clear()
                await self.send_event({"type": "response.create"})
            elif self.is_responding:
                logger.warning("Not sending response.create as another response is in progress")
        except Exception as e:
            logger.error(f"Error sending response.create: {e}")

        # Wait a moment for the server to process
        await asyncio.sleep(0.5)
        print("\nListening mode ended. Type a command to continue.")

        # Return a flag indicating if the user wants to quit
        return quit_requested

    async def manual_record_audio(self):
        """Record audio with visual indicators and manually commit when done."""
        print("Starting manual recording mode...")

        # Wait for any current response to complete
        if self.is_responding:
            print("Assistant is responding. Do you want to interrupt? (y/n)")
            try:
                choice = await asyncio.to_thread(input)
                if choice.lower() != 'y':
                    print("Waiting for current response to complete...")
                    await self.response_complete_event.wait()
                else:
                    print("Interrupting current response...")
                    # Cancel the current response before starting a new one
                    await self.send_event({"type": "response.cancel"})
                    self.reset_audio_state()
                    await asyncio.sleep(0.5)  # Brief pause
            except Exception as e:
                logger.error(f"Error getting interruption choice: {e}")
                # Default to waiting if we can't get input
                await self.response_complete_event.wait()

        # Clear transcript from previous sessions
        self.transcript_processor.clear_transcript()

        self.audio_handler.start_recording()

        # For manual stopping
        stop_recording = False
        quit_requested = False
        recorded_data = b''

        # For transcript display
        transcript_display_task = None

        async def wait_for_input():
            nonlocal stop_recording, quit_requested
            user_input = await asyncio.to_thread(input)
            if user_input.lower() == 'q':
                print("\n=== Quit requested - stopping all processes ===")
                quit_requested = True

                # Set the shutdown flag to prevent any further event processing
                self.shutdown_in_progress = True

                # EMERGENCY STOP: Forcefully terminate all audio processes immediately
                print("EMERGENCY STOP: Forcefully terminating all audio processes...")
                self.audio_handler.emergency_stop()

                # Update the transcript one last time before stopping
                print("Finalizing transcript...")
                self.transcript_processor.update_transcript()

                # Add a small delay to ensure any pending transcript updates are processed
                time.sleep(0.5)

                # Now stop the transcript processor
                print("Stopping transcript processor...")
                self.transcript_processor.stop()

                # Save the transcript immediately
                print("Saving transcript...")
                transcript_file = self.transcript_processor.save_transcript(force=True)
                if transcript_file:
                    print(f"Transcript saved to: {transcript_file}")

                # Clear any buffered audio
                self.audio_buffer = b''

                # Immediately cancel any ongoing response
                if self.is_responding:
                    try:
                        print("Cancelling any ongoing responses...")
                        # Force reset the response state
                        self.is_responding = False
                        self.response_complete_event.set()

                        # Try to cancel the response via WebSocket
                        if self.ws:
                            try:
                                await self.send_event({"type": "response.cancel"})
                            except Exception:
                                # Ignore errors during shutdown
                                pass

                        # Reset audio state
                        self.reset_audio_state()
                    except Exception as e:
                        logger.error(f"Error cancelling response during quit: {e}")
            stop_recording = True

        async def display_transcript_updates():
            """Periodically check for transcript updates (no longer displaying to console)."""
            last_transcript = ""
            while not stop_recording and self.ws:
                # Still check for updates but don't print to console
                current_transcript = self.transcript_processor.get_current_transcript()
                if current_transcript and current_transcript != last_transcript:
                    # Just update the last transcript without printing
                    last_transcript = current_transcript
                await asyncio.sleep(0.5)  # Check every half second

        try:
            # Start the input listener in the background
            enter_task = asyncio.create_task(wait_for_input())

            # Start transcript display task
            transcript_display_task = asyncio.create_task(display_transcript_updates())

            # Keep track of how long we've been recording
            start_time = time.time()
            last_level_time = start_time

            while not stop_recording and self.ws:
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    try:
                        # Analyze audio levels for visual feedback
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        max_amp = np.max(np.abs(audio_array))

                        # Create a visual level meter
                        current_time = time.time()
                        if current_time - last_level_time > 0.1:  # Update 10 times per second
                            # Calculate meter level (0-20)
                            level = min(20, int(max_amp / 500))
                            meter = "[" + "#" * level + "-" * (20 - level) + "]"
                            # Use carriage return to update in place
                            print(f"\rAudio Level: {meter} {'Speaking!' if level > 3 else 'Silent'}", end="", flush=True)
                            last_level_time = current_time

                        # Add to our recorded buffer
                        recorded_data += chunk

                        # Encode and send to the server as we go
                        base64_chunk = base64.b64encode(chunk).decode('utf-8')
                        await self.send_event({
                            "type": "input_audio_buffer.append",
                            "audio": base64_chunk
                        })

                        # --- Send user audio chunk to sentiment analysis ---
                        if self.enable_sentiment_analysis and self.audio_mp_queue:
                            user_audio_chunk = {
                                "source": "user",
                                "data": chunk,
                                "rate": self.audio_handler.rate, # Get rate from handler
                                "timestamp": time.time()
                            }
                            try:
                                self.audio_mp_queue.put_nowait(user_audio_chunk)
                            except queue.Full:
                                logger.warning("Sentiment analysis audio queue is full (manual mode). Skipping chunk.")
                            except Exception as e:
                                logger.error(f"Error sending chunk to sentiment queue (manual mode): {e}")
                        # -------------------------------------------------

                        await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error sending audio chunk: {e}")
                        break
                else:
                    await asyncio.sleep(0.01)

                # Safety timeout - if recording for too long (2 min)
                if current_time - start_time > 120.0:
                    logger.warning("Manual recording timeout reached (2 minutes)")
                    print("\nRecording timeout reached. Stopping automatically.")
                    stop_recording = True

            # Cancel enter_task if it's still running
            if not enter_task.done():
                enter_task.cancel()
                try:
                    await enter_task
                except asyncio.CancelledError:
                    pass

            # Cancel transcript display task
            if transcript_display_task and not transcript_display_task.done():
                transcript_display_task.cancel()
                try:
                    await transcript_display_task
                except asyncio.CancelledError:
                    pass

            print("\nRecording complete, sending to server...")

            # Explicitly commit the audio buffer
            if self.ws and len(recorded_data) > 0:
                # Commit the audio buffer regardless of VAD setting
                await self.send_event({"type": "input_audio_buffer.commit"})

                # Create a response if not already responding
                if not self.is_responding:
                    self.is_responding = True
                    self.response_complete_event.clear()
                    await self.send_event({"type": "response.create"})
                else:
                    logger.warning("Not sending response.create as another response is in progress")

                bytes_recorded = len(recorded_data)
                duration_ms = bytes_recorded / 2 / self.audio_handler.rate * 1000
                print(f"Sent {bytes_recorded} bytes ({duration_ms:.1f} ms) of audio to the server.")
            else:
                print("No audio recorded or connection lost.")

        except Exception as e:
            logger.error(f"Error during manual recording: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Stop recording
            self.audio_handler.stop_recording()
            print("\nManual recording mode ended.")
            await asyncio.sleep(0.5)  # Wait for server processing

            return quit_requested

    async def run(self):
        """Main loop to handle user input and interact with the WebSocket server or chat API."""
        # Check memory integration status
        if self.enable_memory:
            if self.memory_integration:
                print(f"\n[MEMORY SYSTEM] Memory integration is enabled for {self.conversation_mode} mode")
                logger.info(f"Memory integration is enabled for {self.conversation_mode} mode")
            else:
                print("\n[MEMORY SYSTEM] Memory integration is enabled but not initialized properly")
                logger.warning("Memory integration is enabled but not initialized properly")
        else:
            print("\n[MEMORY SYSTEM] Memory integration is disabled")
            logger.info("Memory integration is disabled")

        # Connect to WebSocket only in audio mode
        connection_success = True
        if self.conversation_mode == "audio":
            connection_success = await self.connect()
            if not connection_success:
                logger.error("Failed to establish connection. Exiting...")
                return
        else:
            # In chat mode, we don't need to connect to the WebSocket
            print("Starting in chat mode - using OpenAI Chat Completions API")

        # Start event receiver in background
        receive_task = asyncio.create_task(self.receive_events())

        # Set up signal handling for clean shutdown
        shutdown_event = asyncio.Event()

        # Command options
        if self.conversation_mode == "audio":
            commands = """
            Commands:
            q: Quit and save transcript
            t: Send text message
            a: Start listening mode (press Enter to stop)
            m: Manual recording with visual indicators (press Enter to STOP, better for speech detection)
            l: Start continuous listening (stays active until you press Enter)
            r: Reconnect if WebSocket connection was lost
            s: Store a memory directly
            """
        else:  # chat mode
            commands = """
            Commands:
            q: Quit and save transcript
            r: Reconnect if WebSocket connection was lost
            a: Switch to audio input mode
            s: Store a memory directly

            In chat mode, simply type your message and press Enter to send.
            """

        # Initialize continuous listening mode flag
        continuous_listening = True  # Set to True by default

        try:
            print("\n=============================================")
            print("Connected to OpenAI Realtime API!")
            print("=============================================")
            print("TROUBLESHOOTING TIPS:")
            print("- If you don't hear responses: Make sure your audio device is working")
            print("- If voice detection isn't working: Use 'm' for manual recording mode")
            print("- If the conversation stops: Use 'r' to reconnect")
            print(f"- Transcript will be saved to: {self.transcript_processor.transcripts_dir.absolute()} when you quit")
            print("\nAvailable commands:")
            print(commands)
            if self.conversation_mode == "audio":
                print("Starting in continuous listening mode automatically...")
                print("Press Enter to pause listening when needed.")
            else:  # chat mode
                print("Starting in chat-based conversation mode...")
                print("Type your messages and press Enter to send.")
                continuous_listening = False  # Don't start in listening mode for chat
            print("=============================================\n")

            while not shutdown_event.is_set():
                if continuous_listening:
                    # In continuous listening mode, only check commands occasionally
                    # but keep the audio streaming
                    quit_requested = await self.send_audio()
                    if quit_requested:
                        # User wants to quit during audio streaming
                        # logger.info("Quit requested during continuous listening mode")

                        # Run memory deduplication if memory is enabled
                        if self.enable_memory and self.memory_integration:
                            try:
                                print("\n=== Running memory deduplication ===")
                                duplicates_removed = await self.memory_integration.deduplicate_memories(similarity_threshold=0.7)
                                print(f"Removed {duplicates_removed} duplicate memories")
                                # Wait a moment for deduplication to complete
                                await asyncio.sleep(1)
                            except Exception as e:
                                logger.error(f"Error running memory deduplication: {e}")
                                print(f"Error running memory deduplication: {str(e)}")

                        break

                    print("Continuous listening paused. Type 'l' to resume or 'q' to quit")
                    continuous_listening = False

                # Better input handling
                try:
                    # Use asyncio.to_thread instead of run_in_executor for better cancellation handling
                    prompt = "> " if not self.conversation_mode == "chat" else "You: "
                    command = await asyncio.to_thread(input, prompt)
                except asyncio.CancelledError:
                    # logger.info("Input task cancelled, exiting...")
                    break
                except Exception as e:
                    logger.error(f"Error getting input: {e}")
                    await asyncio.sleep(0.5)
                    continue

                # Process command
                if not command or command.strip() == "":
                    continue

                # In chat mode, treat any input as a message unless it's a quit command
                if self.conversation_mode == "chat":
                    if command.lower() == 'q':
                        print("\n=== Quit command received - stopping all processes ===")

                        # Set the shutdown flag to prevent any further event processing
                        self.shutdown_in_progress = True

                        # EMERGENCY STOP: Forcefully terminate all audio processes immediately
                        print("EMERGENCY STOP: Forcefully terminating all audio processes...")
                        self.audio_handler.emergency_stop()

                        # Update the transcript one last time before stopping
                        print("Finalizing transcript...")
                        self.transcript_processor.update_transcript()

                        # Add a small delay to ensure any pending transcript updates are processed
                        time.sleep(0.5)

                        # Now stop the transcript processor
                        print("Stopping transcript processor...")
                        self.transcript_processor.stop()

                        # Save the transcript immediately
                        print("Saving transcript...")
                        transcript_file = self.transcript_processor.save_transcript(force=True)
                        if transcript_file:
                            print(f"Transcript saved to: {transcript_file}")

                        # Clear any buffered audio
                        self.audio_buffer = b''

                        # Immediately cancel any ongoing response
                        if self.is_responding:
                            try:
                                print("Cancelling any ongoing responses...")
                                # Force reset the response state
                                self.is_responding = False
                                self.response_complete_event.set()

                                # In chat mode, we don't have a WebSocket, so just reset the state
                                self.reset_audio_state()
                            except Exception as e:
                                logger.error(f"Error cancelling response during quit: {e}")

                        # Run memory deduplication if memory is enabled
                        if self.enable_memory and self.memory_integration:
                            try:
                                print("\n=== Running memory deduplication ===")
                                duplicates_removed = await self.memory_integration.deduplicate_memories(similarity_threshold=0.7)
                                print(f"Removed {duplicates_removed} duplicate memories")
                                # Wait a moment for deduplication to complete
                                await asyncio.sleep(1)
                            except Exception as e:
                                logger.error(f"Error running memory deduplication: {e}")
                                print(f"Error running memory deduplication: {str(e)}")

                        # Exit the loop immediately
                        break
                    else:
                        # Send the text as a message
                        await self.send_text(command)
                        continue

                if command.lower() == 'q':
                    # logger.info("Quit command received - exiting application")
                    print("\n=== Quit command received - stopping all processes ===")

                    # Set the shutdown flag to prevent any further event processing
                    self.shutdown_in_progress = True

                    # EMERGENCY STOP: Forcefully terminate all audio processes immediately
                    print("EMERGENCY STOP: Forcefully terminating all audio processes...")
                    self.audio_handler.emergency_stop()

                    # Update the transcript one last time before stopping
                    print("Finalizing transcript...")
                    self.transcript_processor.update_transcript()

                    # Add a small delay to ensure any pending transcript updates are processed
                    time.sleep(0.5)

                    # Now stop the transcript processor
                    print("Stopping transcript processor...")
                    self.transcript_processor.stop()

                    # Save the transcript immediately
                    print("Saving transcript...")
                    transcript_file = self.transcript_processor.save_transcript(force=True)
                    if transcript_file:
                        print(f"Transcript saved to: {transcript_file}")

                    # Clear any buffered audio
                    self.audio_buffer = b''

                    # Immediately cancel any ongoing response
                    if self.is_responding:
                        try:
                            print("Cancelling any ongoing responses...")
                            # Force reset the response state
                            self.is_responding = False
                            self.response_complete_event.set()

                            # Try to cancel the response via WebSocket
                            if self.ws:
                                try:
                                    await self.send_event({"type": "response.cancel"})
                                except Exception:
                                    # Ignore errors during shutdown
                                    pass

                            # Reset audio state
                            self.reset_audio_state()
                        except Exception as e:
                            logger.error(f"Error cancelling response during quit: {e}")

                    # Run memory deduplication if memory is enabled
                    if self.enable_memory and self.memory_integration:
                        try:
                            print("\n=== Running memory deduplication ===")
                            duplicates_removed = await self.memory_integration.deduplicate_memories(similarity_threshold=0.7)
                            print(f"Removed {duplicates_removed} duplicate memories")
                            # Wait a moment for deduplication to complete
                            await asyncio.sleep(1)
                        except Exception as e:
                            logger.error(f"Error running memory deduplication: {e}")
                            print(f"Error running memory deduplication: {str(e)}")

                    break  # Exit the loop immediately

                elif command.lower() == 't':
                    # Check if WebSocket exists
                    if not self.ws:
                        logger.error("WebSocket connection is not established. Trying to reconnect...")
                        connection_success = await self.connect()
                        if not connection_success:
                            logger.error("Failed to reconnect. Please try again.")
                            continue

                    try:
                        # Get text input with better error handling
                        print("Enter your message (press Enter to send):")
                        text = await asyncio.to_thread(input)
                        if text.strip():
                            await self.send_text(text)
                        else:
                            print("Empty message, not sending.")
                    except Exception as e:
                        logger.error(f"Error sending text: {e}")

                elif command.lower() == 'a':
                    # Start single listening session
                    # Check if WebSocket exists
                    if not self.ws:
                        logger.error("WebSocket connection is not established. Trying to reconnect...")
                        connection_success = await self.connect()
                        if not connection_success:
                            logger.error("Failed to reconnect. Please try again.")
                            continue

                    print("Starting listening mode... (press Enter to stop)")
                    try:
                        await self.send_audio()
                    except Exception as e:
                        logger.error(f"Error in listening mode: {e}")

                elif command.lower() == 'm':
                    # Manual recording with visual indicators (better for speech detection)
                    if not self.ws:
                        logger.error("WebSocket connection is not established. Trying to reconnect...")
                        connection_success = await self.connect()
                        if not connection_success:
                            logger.error("Failed to reconnect. Please try again.")
                            continue

                    print("\n=== MANUAL RECORDING MODE ===")
                    print("Speak now! Recording will show audio levels.")
                    print("Press Enter when you're DONE speaking to send the recording.")
                    print("Or press 'q' to quit the application.")
                    print("Audio Level: [-------------------]")
                    try:
                        quit_requested = await self.manual_record_audio()
                        if quit_requested:
                            # User wants to quit during manual recording
                            # logger.info("Quit requested during manual recording mode")

                            print("\n=== Quit command received - stopping all processes ===")

                            # Set the shutdown flag to prevent any further event processing
                            self.shutdown_in_progress = True

                            # EMERGENCY STOP: Forcefully terminate all audio processes immediately
                            print("EMERGENCY STOP: Forcefully terminating all audio processes...")
                            self.audio_handler.emergency_stop()

                            # Update the transcript one last time before stopping
                            print("Finalizing transcript...")
                            self.transcript_processor.update_transcript()

                            # Add a small delay to ensure any pending transcript updates are processed
                            time.sleep(0.5)

                            # Now stop the transcript processor
                            print("Stopping transcript processor...")
                            self.transcript_processor.stop()

                            # Save the transcript immediately
                            print("Saving transcript...")
                            transcript_file = self.transcript_processor.save_transcript(force=True)
                            if transcript_file:
                                print(f"Transcript saved to: {transcript_file}")

                            # Clear any buffered audio
                            self.audio_buffer = b''

                            # Immediately cancel any ongoing response
                            if self.is_responding:
                                try:
                                    print("Cancelling any ongoing responses...")
                                    # Force reset the response state
                                    self.is_responding = False
                                    self.response_complete_event.set()

                                    # Try to cancel the response via WebSocket
                                    if self.ws:
                                        try:
                                            await self.send_event({"type": "response.cancel"})
                                        except Exception:
                                            # Ignore errors during shutdown
                                            pass

                                    # Reset audio state
                                    self.reset_audio_state()
                                except Exception as e:
                                    logger.error(f"Error cancelling response during quit: {e}")

                            # Run memory deduplication if memory is enabled
                            if self.enable_memory and self.memory_integration:
                                try:
                                    print("\n=== Running memory deduplication ===")
                                    duplicates_removed = await self.memory_integration.deduplicate_memories(similarity_threshold=0.7)
                                    print(f"Removed {duplicates_removed} duplicate memories")
                                    # Wait a moment for deduplication to complete
                                    await asyncio.sleep(1)
                                except Exception as e:
                                    logger.error(f"Error running memory deduplication: {e}")
                                    print(f"Error running memory deduplication: {str(e)}")

                            break
                    except Exception as e:
                        logger.error(f"Error in manual recording: {e}")

                elif command.lower() == 'l':
                    # Start continuous listening mode
                    if not self.ws:
                        logger.error("WebSocket connection is not established. Trying to reconnect...")
                        connection_success = await self.connect()
                        if not connection_success:
                            logger.error("Failed to reconnect. Please try again.")
                            continue

                    print("Starting continuous listening mode... (session will continue until you press Enter)")
                    continuous_listening = True

                elif command.lower() == 'r':
                    # Force reconnection
                    # logger.info("Manually reconnecting to the server...")
                    if self.ws:
                        try:
                            await self.ws.close()
                        except:
                            pass
                        self.ws = None

                    connection_success = await self.connect()
                    if connection_success:
                        print("Successfully reconnected to the server!")
                    else:
                        print("Failed to reconnect. Please try again.")

                elif command.lower() == 's':
                    # Store a memory directly
                    if not self.enable_memory or not self.memory_integration:
                        print("Memory system is not enabled.")
                        continue

                    try:
                        print("Enter the memory content:")
                        content = await asyncio.to_thread(input)
                        if not content.strip():
                            print("Empty content, not storing.")
                            continue

                        print("Enter the importance (1-10, default 8):")
                        importance_str = await asyncio.to_thread(input)
                        importance = 8
                        if importance_str.strip():
                            try:
                                importance = int(importance_str)
                                if importance < 1 or importance > 10:
                                    print("Importance must be between 1 and 10, using default 8.")
                                    importance = 8
                            except ValueError:
                                print("Invalid importance, using default 8.")

                        print("Enter the category (default 'preference'):")
                        category = await asyncio.to_thread(input)
                        if not category.strip():
                            category = "preference"

                        # Store the memory directly
                        result = await self.memory_integration.store_memory_directly(
                            content=content,
                            importance=importance,
                            category=category
                        )

                        print(f"Memory storage result: {result}")
                    except Exception as e:
                        logger.error(f"Error storing memory directly: {e}")
                        print(f"Error storing memory: {str(e)}")

                else:
                    print(f"Unknown command: {command}")
                    print(commands)

                # Small delay to allow other tasks to run
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # Update the transcript one last time before saving
            print("Finalizing transcript...")
            self.transcript_processor.update_transcript()

            # Add a small delay to ensure any pending transcript updates are processed
            time.sleep(0.5)

            # Save transcript before exiting (force=True to ensure it's saved)
            print("Saving transcript and exiting...")
            transcript_file = self.transcript_processor.save_transcript(force=True)
            if transcript_file:
                print(f"Complete transcript saved to: {transcript_file}")
            else:
                print("No transcript to save.")

            # --- Retrieve and print sentiment results ---
            if self.enable_sentiment_analysis and self.sentiment_history_list is not None:
                 # Convert shared list to regular list for printing
                 final_sentiment_results = list(self.sentiment_history_list)
                 if final_sentiment_results:
                      print("\n--- Sentiment Analysis Results ---")
                      for result in final_sentiment_results:
                          if 'error' in result:
                               ts_err = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                               print(f"[{ts_err}] Sentiment Analysis ERROR: {result['error']}")
                               if 'traceback' in result:
                                    print(result['traceback'])
                          else:
                              ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                              duration = result.get('audio_duration_s', 'N/A')
                              label = result.get('sentiment_label', 'N/A')
                              value = result.get('sentiment_value', 'N/A')
                              raw = result.get('raw_response', '')
                              # Format duration safely
                              duration_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else str(duration)
                              print(f"[{ts}] Label: {label} ({value}), Duration: {duration_str}, Raw: '{raw}'")
                      print("----------------------------------")
                 else:
                      print("\nNo sentiment analysis results were generated.")
            # -------------------------------------------

            # Set shutdown event to signal all tasks to stop
            shutdown_event.set()

            # Cancel receive task
            if not receive_task.done():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling receive task: {e}")

            # Cleanup
            await self.cleanup()
            # logger.info("Shutdown complete")

            # Return from the run method to allow proper cleanup
            print("Application terminated.")

    async def cleanup(self):
        """Clean up resources."""
        # logger.info("Cleaning up resources")
        self.audio_handler.cleanup()

        # Stop the transcript processor
        self.transcript_processor.stop()

        if self.ws:
            try:
                await self.ws.close()
                # logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
                # Even if closing fails, set to None to prevent further usage
            self.ws = None

        # --- Stop the sentiment analysis process ---
        if self.sentiment_process and self.sentiment_process.is_alive():
            logger.info("Stopping sentiment analysis process...")
            try:
                self.sentiment_process.stop() # Signal it to stop
                # Wait briefly for it to exit gracefully
                self.sentiment_process.join(timeout=5)
                if self.sentiment_process.is_alive():
                    logger.warning("Sentiment process did not exit gracefully, terminating...")
                    self.sentiment_process.terminate() # Force terminate if needed
                    self.sentiment_process.join(timeout=1) # Wait for termination
                else:
                     logger.info("Sentiment analysis process stopped.")
            except Exception as e:
                logger.error(f"Error stopping sentiment analysis process: {e}")

            # Stop the sentiment update thread
            self.sentiment_stop_event.set()
            if self.sentiment_update_thread and self.sentiment_update_thread.is_alive():
                logger.info("Waiting for sentiment update thread to stop...")
                self.sentiment_update_thread.join(timeout=2)
                if self.sentiment_update_thread.is_alive():
                     logger.warning("Sentiment update thread did not stop gracefully.")
            # -------------------------------------------
        # ------------------------------------------

        # --- Shutdown multiprocessing manager ---
        if self.manager:
             logger.info("Shutting down multiprocessing manager...")
             try:
                 self.manager.shutdown()
             except Exception as e:
                 logger.error(f"Error shutting down multiprocessing manager: {e}")
             self.manager = None # Ensure it's cleared
        # ---------------------------------------

        # --- Clean up memory integration ---
        if self.enable_memory and self.memory_integration:
            try:
                await self.memory_integration.cleanup()
                logger.info("Memory integration cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up memory integration: {e}")
            self.memory_integration = None
        # ---------------------------------------

    def reset_audio_state(self):
        """Reset audio state when interrupted."""
        if self.is_responding:
            # logger.info("Resetting audio state due to interruption")
            self.is_responding = False
            self.response_complete_event.set()
            self.audio_buffer = b''  # Clear any buffered audio

            # Use emergency stop to forcefully terminate all audio processes
            self.audio_handler.emergency_stop()

            # Reset current response text if in chat mode
            if self.conversation_mode == "chat" and hasattr(self, "current_response_text"):
                self.current_response_text = ""

            # Reset current audio response text if in audio mode
            if self.conversation_mode == "audio" and hasattr(self, "current_audio_response_text"):
                self.current_audio_response_text = ""

            print("\nResponse interrupted.")

    def _continuously_update_sentiment(self):
        """Runs in a separate thread to periodically check for new sentiment results and update the shared dict."""
        last_processed_index = -1
        print("[Sentiment Updater] Thread started.")
        
        # Initialize sentiment results list
        sentiment_results = []
        
        # Get the transcripts directory path and create timestamped filename
        transcripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gpt_realtime', 'gpt_transcripts')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sentiment_file_path = os.path.join(transcripts_dir, f'sentiment_results_{timestamp}.json')
        
        while not self.sentiment_stop_event.is_set():
            try:
                # Access shared list carefully
                if self.sentiment_history_list is not None and self.shared_emotion_scores is not None:
                    current_len = len(self.sentiment_history_list)
                    if current_len > last_processed_index + 1:
                        # Process new items
                        new_results = self.sentiment_history_list[last_processed_index + 1:]
                        print(f"[Sentiment Updater] Found {len(new_results)} new sentiment results.")
                        
                        for result in new_results:
                            if isinstance(result, dict):
                                # Add timestamp if not present
                                if 'timestamp' not in result:
                                    result['timestamp'] = time.time()
                                
                                # Parse emotion scores from raw response if needed
                                if 'emotion_scores' not in result and 'raw_response' in result:
                                    try:
                                        raw = result['raw_response']
                                        # Clean up the raw response if it contains markdown
                                        if '```json' in raw:
                                            raw = raw.split('```json')[1].split('```')[0].strip()
                                        emotion_scores = json.loads(raw)
                                        result['emotion_scores'] = emotion_scores
                                    except json.JSONDecodeError as e:
                                        print(f"[Sentiment Updater] Error parsing emotion scores: {e}")
                                        continue
                                
                                # Add to sentiment results
                                sentiment_results.append(result)
                                
                                # Update the shared dictionary atomically
                                if 'emotion_scores' in result:
                                    self.shared_emotion_scores.update(result['emotion_scores'])
                        
                        # Save updated results to JSON file in transcripts directory
                        try:
                            with open(sentiment_file_path, 'w') as f:
                                json.dump(sentiment_results, f, indent=2)
                            print(f"[Sentiment Updater] Saved {len(sentiment_results)} sentiment results to {sentiment_file_path}")
                        except Exception as e:
                            print(f"[Sentiment Updater] Error saving sentiment results: {e}")
                        
                        last_processed_index = current_len - 1

                # Sleep for a short duration
                time.sleep(0.5)  # Check twice per second

            except Exception as e:
                print(f"[Sentiment Updater] Error in update loop: {e}")
                traceback.print_exc()
                # Avoid tight loop on error
                time.sleep(2)

        # Final save before thread stops
        try:
            with open(sentiment_file_path, 'w') as f:
                json.dump(sentiment_results, f, indent=2)
            print(f"[Sentiment Updater] Final save: {len(sentiment_results)} sentiment results to {sentiment_file_path}")
        except Exception as e:
            print(f"[Sentiment Updater] Error in final save: {e}")

        print("[Sentiment Updater] Thread stopped.")

# --- Dash App Setup ---
# Define the Dash app globally so it can be accessed by the callback
app = dash.Dash(__name__)

# Initial empty data for the chart
initial_scores = {emotion: 0 for emotion in CORE_EMOTIONS}
initial_radar_data = format_data_for_radar(initial_scores)

# Define the layout
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    withGlobalClasses=True,
    children=[
        html.Div([
            dmc.Title("Real-time Emotion Analysis", order=2, ta="center"),
            dmc.Text("(Based on User Input Audio)", ta="center", size="sm", c="dimmed"),
            dmc.Space(h=20),
            dmc.RadarChart(
                id="emotion-radar-chart",
                h=450, # Height of the chart
                data=initial_radar_data, # Initial data
                dataKey="emotion", # Key in data dictionaries for axis labels
                withPolarGrid=True,
                withPolarAngleAxis=True,
                withPolarRadiusAxis=True,
                polarRadiusAxisProps={"domain": [0, 3], "tickCount": 4}, # Renamed radiusAxisProps
                series=[
                    # Define the data series to plot
                    {"name": "Sentiment", "color": "indigo.6", "opacity": 0.7}
                ],
            ),
            dcc.Interval(
                id='interval-component',
                interval=500, # Update every 0.5 seconds (500 milliseconds)
                n_intervals=0
            )
        ], style={"maxWidth": "600px", "margin": "auto", "padding": "20px"})
    ]
)

# Define the callback to update the chart (needs access to shared_emotion_scores)
# We will pass the shared dictionary to this function within main
def setup_callbacks(shared_scores):
    @app.callback(
        Output('emotion-radar-chart', 'data'),
        Input('interval-component', 'n_intervals')
    )
    def update_radar_chart(_):
        """Reads from the shared dictionary and updates the chart data."""
        if shared_scores is not None:
            # Read the current scores (convert Manager.dict to regular dict)
            current_scores = dict(shared_scores)
            # Format the data for the radar chart
            new_radar_data = format_data_for_radar(current_scores)
            # print(f"[Dash Callback] Updating chart with data: {new_radar_data}") # Debug
            return new_radar_data
        else:
            # Return empty data or default if shared_scores not available
            return format_data_for_radar({emotion: 0 for emotion in CORE_EMOTIONS})
# --- End Dash App Setup ---

async def handle_user_account():
    """Handle user login or registration at startup.

    Returns:
        Tuple of (user_id, username) if successful, (None, None) otherwise
    """
    print("\n=== User Account System ===")
    print("Please enter your username to log in or register a new account.")
    username = input("Username: ").strip()

    if not username:
        print("No username provided. Continuing as anonymous user.")
        return None, None

    # Initialize user account manager
    user_manager = UserAccountManager()
    if not user_manager.connect():
        logger.error("Failed to connect to Supabase for user account management")
        print("Error connecting to user database. Continuing as anonymous user.")
        return None, None

    # Try to log in with the provided username
    user = user_manager.login_user(username)

    if user:
        print(f"\nWelcome back, {user.get('display_name', username)}!")
        return user.get('id'), username
    else:
        # User doesn't exist, ask if they want to register
        print(f"\nUser '{username}' not found.")
        register = input("Would you like to register a new account? (y/n): ").strip().lower()

        if register == 'y':
            # Ask for optional display name
            display_name = input("Display name (optional, press Enter to use username): ").strip()
            if not display_name:
                display_name = username

            # Register the new user
            user = user_manager.register_user(username=username, display_name=display_name)

            if user:
                print(f"\nAccount created successfully. Welcome, {display_name}!")
                return user.get('id'), username
            else:
                print("\nFailed to create account. Continuing as anonymous user.")
                return None, None
        else:
            print("\nContinuing as anonymous user.")
            return None, None

async def main(conversation_mode="audio", enable_memory=True):
    # Initialize user variables
    user_id = None
    username = None

    # Load API keys from file
    openai_api_key = None
    gemini_api_key = None
    api_key_path = '.api_key.json' # Default path
    try:
        # Look for the key file in the parent directory relative to this script
        script_dir = os.path.dirname(__file__)
        potential_path = os.path.join(script_dir, '..', '.api_key.json')
        if os.path.exists(potential_path):
             api_key_path = potential_path
        elif not os.path.exists(api_key_path): # Check default only if parent doesn't exist
             logger.error(f"API key file not found at {os.path.abspath(potential_path)} or {os.path.abspath(api_key_path)}")
             return

        logger.info(f"Attempting to load API keys from: {os.path.abspath(api_key_path)}")

        with open(api_key_path, "r") as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get("openai_api_key")
            gemini_api_key = api_keys.get("gemini_api_key") # Key for Google Gemini
    except FileNotFoundError:
        logger.error(f"API key file not found at {os.path.abspath(api_key_path)} or fallback location.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from API key file: {os.path.abspath(api_key_path)}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading API keys: {e}")
        return

    if not openai_api_key:
        logger.error("OpenAI API key ('openai_api_key') not found in .api_key.json")
        return

    # --- Handle User Account --- #
    user_id, username = await handle_user_account()

    # Decide whether to enable sentiment analysis
    # For now, enable it if the Gemini key is present and the class was imported
    enable_sentiment = bool(gemini_api_key and SentimentAnalysisProcess)
    if enable_sentiment:
        logger.info("Gemini API key found. Enabling sentiment analysis.")
        print("\nSentiment analysis is ENABLED.")
    elif gemini_api_key:
        logger.warning("Gemini API key found, but SentimentAnalysisProcess class not loaded. Sentiment analysis disabled.")
        print("\nSentiment analysis is DISABLED: SentimentAnalysisProcess class not loaded.")
    else:
        logger.info("Gemini API key not found. Sentiment analysis disabled.")
        print("\nSentiment analysis is DISABLED: Gemini API key not found.")

    # --- Multiprocessing and Shared Data Setup ---
    manager = None
    shared_emotion_scores = None
    if enable_sentiment:
        try:
            manager = multiprocessing.Manager()
            # Initialize shared dict with default scores
            shared_emotion_scores = manager.dict({emotion: 0 for emotion in CORE_EMOTIONS})
            logger.info("Multiprocessing manager and shared emotion dictionary initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize multiprocessing manager or shared dict: {e}")
            logger.warning("Disabling sentiment analysis due to manager initialization failure.")
            enable_sentiment = False # Disable if manager fails
            # Ensure cleanup if partially initialized
            if manager:
                try: manager.shutdown()
                except Exception: pass
            manager = None
            shared_emotion_scores = None
    # ---------------------------------------------

    # Create client, passing the shared dictionary if sentiment analysis is enabled
    client = RealtimeClient(
        api_key=openai_api_key,
        google_api_key=gemini_api_key,
        enable_sentiment_analysis=enable_sentiment, # Use potentially updated value
        shared_emotion_scores=shared_emotion_scores, # Pass the shared dict (or None)
        conversation_mode=conversation_mode, # Pass the conversation mode
        enable_memory=enable_memory, # Pass the memory flag
        user_id=user_id, # Pass the user ID
        username=username # Pass the username
    )

    # Print the selected mode and memory status
    logger.info(f"Starting in {conversation_mode} conversation mode")
    logger.info(f"Memory system is {'enabled' if enable_memory else 'disabled'}")

    # Setup Dash callbacks, passing the shared dictionary
    if enable_sentiment:
        setup_callbacks(shared_emotion_scores)
    else:
        # If sentiment is disabled, setup callbacks with None
        # The callback will just show default 0 values
        setup_callbacks(None)
        logger.info("Dash callbacks set up with default data (sentiment analysis disabled).")

    # --- Threading Setup for Client ---
    client_thread = None

    async def run_client_async():
        nonlocal client
        try:
            await client.run()
        except Exception as e:
             logger.error(f"Error in RealtimeClient run: {e}")
             traceback.print_exc()
        finally:
            logger.info("RealtimeClient run has completed or exited.")
            # Note: client.cleanup() is called within client.run()'s finally block

    # Start the client in a separate thread
    # Running an async function in a thread requires care.
    # We need an event loop in that thread.
    def start_client_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_client_async())
        finally:
            loop.close()
            logger.info("Client thread event loop closed.")

    # Use daemon=False so we can properly join this thread
    client_thread = threading.Thread(target=start_client_thread, daemon=False)
    client_thread.start()
    logger.info("RealtimeClient started in a background thread.")
    # -----------------------------------

    # --- Create a shutdown event for Dash app --- #
    # Create a threading event to signal when to shut down the Dash app
    dash_shutdown_event = threading.Event()

    # Create a function to check if we should shut down the Dash app
    def check_shutdown():
        if dash_shutdown_event.is_set():
            # If the shutdown event is set, shut down the Dash server
            logger.info("Shutting down Dash server...")
            import flask
            # Get the Flask server from the Dash app
            # We need to access app.server to initialize the Flask server
            # even though we don't use the variable directly
            _ = app.server
            # Request shutdown of the Flask server
            func = flask.request.environ.get('werkzeug.server.shutdown')
            if func is None:
                logger.warning("Not running with Werkzeug server, cannot shut down cleanly")
                # Force exit as fallback
                import os
                os._exit(0)
            func()
            return True
        # Check again in 1 second
        import dash
        dash.callback_context.record_timing("shutdown-check", 1000, check_shutdown)
        return False

    # Add a clientside callback to check for shutdown
    app.clientside_callback(
        """
        function(n_intervals) {
            return window.dash_clientside.no_update;
        }
        """,
        dash.Output('interval-component', 'disabled', allow_duplicate=True),
        dash.Input('interval-component', 'n_intervals'),
        prevent_initial_call=True
    )

    # Add a server-side callback to check for shutdown
    @app.callback(
        dash.Output('interval-component', 'disabled'),
        dash.Input('interval-component', 'n_intervals'),
        prevent_initial_call=True
    )
    def check_for_shutdown(_):
        # Underscore parameter name to indicate it's not used
        return check_shutdown()

    # --- Run Dash App --- #
    # Start Dash in a separate thread so it doesn't block
    dash_thread = threading.Thread(target=lambda: app.run(debug=False, host='0.0.0.0', port=8050), daemon=True)
    dash_thread.start()
    print("Starting Dash server for Emotion Radar Chart...")
    print("Access the chart at: http://127.0.0.1:8050/")

    try:
        # Wait for the client thread to finish
        client_thread.join()
        logger.info("Client thread has exited, signaling Dash to shut down")
        # Signal Dash to shut down
        dash_shutdown_event.set()
        # Give Dash a moment to shut down gracefully
        dash_thread.join(timeout=5)
        if dash_thread.is_alive():
            logger.warning("Dash thread did not exit gracefully, forcing shutdown")
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
        # Signal Dash to shut down
        dash_shutdown_event.set()
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
        traceback.print_exc()
    finally:
        logger.info("Application shutting down.")
        # Ensure manager is shutdown if it was created
        if manager:
            logger.info("Attempting to shut down multiprocessing manager...")
            try:
                manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down multiprocessing manager: {e}")

        logger.info("Exiting application main function.")
    # ------------------ #


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('websockets.client').setLevel(logging.WARNING)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='OpenAI Realtime API Client')
    parser.add_argument('--mode', type=str, choices=['audio', 'chat'], default='audio',
                        help='Conversation mode: audio (default) or chat')
    parser.add_argument('--memory', action='store_true', help='Enable memory system (default)')
    parser.add_argument('--no-memory', action='store_true', help='Disable memory system')
    args = parser.parse_args()

    # Determine if memory should be enabled
    enable_memory = not args.no_memory if args.no_memory else True

    # Pass the conversation mode to  main
    asyncio.run(main(conversation_mode=args.mode, enable_memory=enable_memory))