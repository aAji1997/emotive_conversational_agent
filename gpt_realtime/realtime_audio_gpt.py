from transcript_and_audio import AudioHandler, TranscriptProcessor
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
import os
import traceback

# Dash / Plotting Imports
import dash
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output

# Attempt to import sentiment analyzer, handle gracefully if it doesn't exist
try:
    from sentiment_analyzer import SentimentAnalysisProcess
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
    def __init__(self, api_key, google_api_key=None, voice="alloy", enable_sentiment_analysis=False, shared_emotion_scores=None):
        # WebSocket Configuration
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-mini-realtime-preview-2024-12-17"
        self.api_key = api_key # OpenAI API Key
        self.voice = voice
        self.ws = None
        self.audio_handler = AudioHandler()
        
        self.audio_buffer = b''  # Buffer for streaming audio responses
        
        # Add flag to track if assistant is currently responding
        self.is_responding = False
        self.response_complete_event = asyncio.Event()
        self.response_complete_event.set()  # Initially set to True (not responding)

        # Create transcript processor
        self.transcript_processor = TranscriptProcessor()
        
        # Flag to track if we received a transcript for current audio session
        self.received_transcript = False

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
        self.session_config = {
            "modalities": ["audio", "text"],
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "instructions": "You are a helpful AI assistant. Respond in a clear and engaging way. If the user's audio is quiet or unclear, still do your best to respond appropriately.",
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
                print(event["delta"], end="", flush=True)
                
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

                self.audio_handler.play_audio(self.audio_buffer)
                self.audio_buffer = b'' # Clear buffer AFTER sending and playing
            
            print("Saving transcript and exiting...")
            transcript_file = self.transcript_processor.save_transcript()
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

            # Signal that the response is complete
            self.is_responding = False
            self.response_complete_event.set()
            print("\nResponse complete. You can speak again or type a command.")
            
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
        """Send a text message to the WebSocket server."""
        # logger.info(f"Sending text message: {text}")
        
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
        
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
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
                quit_requested = True
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
                quit_requested = True
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
        """Main loop to handle user input and interact with the WebSocket server."""
        # Connect to WebSocket
        connection_success = await self.connect()
        if not connection_success:
            logger.error("Failed to establish connection. Exiting...")
            return
        
        # Start event receiver in background
        receive_task = asyncio.create_task(self.receive_events())
        
        # Set up signal handling for clean shutdown
        shutdown_event = asyncio.Event()

        # Command options
        commands = """
        Commands:
        q: Quit and save transcript
        t: Send text message
        a: Start listening mode (press Enter to stop)
        m: Manual recording with visual indicators (press Enter to STOP, better for speech detection)
        l: Start continuous listening (stays active until you press Enter)
        r: Reconnect if WebSocket connection was lost
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
            print("Starting in continuous listening mode automatically...")
            print("Press Enter to pause listening when needed.")
            print("=============================================\n")
            
            while not shutdown_event.is_set():
                if continuous_listening:
                    # In continuous listening mode, only check commands occasionally
                    # but keep the audio streaming
                    quit_requested = await self.send_audio()
                    if quit_requested:
                        # User wants to quit during audio streaming
                        # logger.info("Quit requested during continuous listening mode")
                        break
                    
                    print("Continuous listening paused. Type 'l' to resume or 'q' to quit")
                    continuous_listening = False
                
                # Better input handling
                try:
                    # Use asyncio.to_thread instead of run_in_executor for better cancellation handling
                    command = await asyncio.to_thread(input, "> ")
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
                    
                if command.lower() == 'q':
                    # logger.info("Quit command received - exiting application")
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
            # Save transcript before exiting
            print("Saving transcript and exiting...")
            transcript_file = self.transcript_processor.save_transcript()
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
            
            # Exit the application immediately
            print("Application terminated.")
            import sys
            sys.exit(0)

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

    def reset_audio_state(self):
        """Reset audio state when interrupted."""
        if self.is_responding:
            # logger.info("Resetting audio state due to interruption")
            self.is_responding = False
            self.response_complete_event.set()
            self.audio_buffer = b''  # Clear any buffered audio
            # Stop any ongoing audio playback
            self.audio_handler.stop_playback()
            print("\nResponse interrupted.")

    def _continuously_update_sentiment(self):
        """Runs in a separate thread to periodically check for new sentiment results and update the shared dict."""
        last_processed_index = -1
        print("[Sentiment Updater] Thread started.")
        while not self.sentiment_stop_event.is_set():
            try:
                # Access shared list carefully
                if self.sentiment_history_list is not None and self.shared_emotion_scores is not None:
                    current_len = len(self.sentiment_history_list)
                    if current_len > last_processed_index + 1:
                        # Process new items
                        new_results = self.sentiment_history_list[last_processed_index + 1:]
                        # print(f"[Sentiment Updater] Found {len(new_results)} new sentiment results.") # Debug
                        for result in new_results:
                            if isinstance(result, dict) and 'emotion_scores' in result:
                                latest_scores = result['emotion_scores']
                                # Update the shared dictionary atomically (Manager.dict handles this)
                                self.shared_emotion_scores.update(latest_scores)
                                # print(f"[Sentiment Updater] Updated shared scores: {dict(self.shared_emotion_scores)}") # Debug
                        last_processed_index = current_len - 1

                # Sleep for a short duration
                time.sleep(0.5) # Check twice per second

            except Exception as e:
                 print(f"[Sentiment Updater] Error in update loop: {e}")
                 traceback.print_exc() # Use traceback for full error details
                 # Avoid tight loop on error
                 time.sleep(2)

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
    def update_radar_chart(n):
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

async def main():
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

    # Decide whether to enable sentiment analysis
    # For now, enable it if the Gemini key is present and the class was imported
    enable_sentiment = bool(gemini_api_key and SentimentAnalysisProcess)
    if enable_sentiment:
        logger.info("Gemini API key found. Enabling sentiment analysis.")
    elif gemini_api_key:
        logger.warning("Gemini API key found, but SentimentAnalysisProcess class not loaded. Sentiment analysis disabled.")
    else:
        logger.info("Gemini API key not found. Sentiment analysis disabled.")

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
        shared_emotion_scores=shared_emotion_scores # Pass the shared dict (or None)
    )

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
    client_stop_event = asyncio.Event() # Use asyncio event if needed within async context, though cleanup is handled internally

    async def run_client_async():
        nonlocal client
        try:
            await client.run()
        except Exception as e:
             logger.error(f"Error in RealtimeClient run: {e}")
             traceback.print_exc()
        finally:
            logger.info("RealtimeClient run has completed or exited.")
            # Signal main thread or handle cleanup if necessary
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

    client_thread = threading.Thread(target=start_client_thread, daemon=True)
    client_thread.start()
    logger.info("RealtimeClient started in a background thread.")
    # -----------------------------------

    # --- Run Dash App --- #
    # Run Dash in the main thread. It will block here.
    print("Starting Dash server for Emotion Radar Chart...")
    print("Access the chart at: http://127.0.0.1:8050/")
    try:
        # Host 0.0.0.0 makes it accessible on the network
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        logger.info("Dash server interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Error running Dash server: {e}")
        traceback.print_exc()
    finally:
        logger.info("Dash server shutting down.")
        # Signal the client thread to stop (client.run handles internal cleanup)
        # The client's cleanup should already be triggered by its own shutdown logic
        # or when the main program exits (due to daemon thread).
        # Ensure manager is shutdown if it was created
        if manager:
            logger.info("Attempting to shut down multiprocessing manager...")
            try:
                manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down multiprocessing manager: {e}")

        # Wait briefly for the client thread to finish its cleanup if needed
        if client_thread and client_thread.is_alive():
             logger.info("Waiting briefly for client thread cleanup...")
             client_thread.join(timeout=5)
             if client_thread.is_alive():
                 logger.warning("Client thread did not exit after Dash shutdown.")

        logger.info("Exiting application main function.")
    # ------------------ #


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('websockets.client').setLevel(logging.WARNING)

    asyncio.run(main()) 