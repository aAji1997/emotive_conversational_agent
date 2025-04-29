import multiprocessing
import queue
import threading
import time
import traceback
import json
import tempfile
import wave
import os
import logging

# Completely disable all logging for this module
logging.getLogger('gpt_realtime.sentiment_analyzer').setLevel(logging.CRITICAL)
logging.getLogger('gpt_realtime.shared_emotion_state').setLevel(logging.CRITICAL)
logging.getLogger('__main__').setLevel(logging.CRITICAL)

# Disable all print statements by redefining print
original_print = print
def silent_print(*args, **kwargs):
    pass
print = silent_print

# Import the SharedEmotionState class
from gpt_realtime.shared_emotion_state import SharedEmotionState
try:
    from google import genai
except ImportError:
    print("ERROR: The 'google-genai' library is required for sentiment analysis.")
    print("Please install it using: pip install google-genai")

    raise ImportError("Missing dependency: google-genai")

# Define the core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

logger = logging.getLogger(__name__)


class SentimentAnalysisProcess(multiprocessing.Process):
    """
    A separate process for performing sentiment analysis on audio chunks using Gemini.
    This runs independently to avoid blocking the main audio processing loop.
    Audio chunks (dictionaries with 'source', 'data', 'rate') are received via a queue.
    Results (dictionaries with 'timestamp', 'sentiment_value', etc.) are placed in a shared list.
    """

    def __init__(self, api_key, sentiment_model_name, audio_queue, sentiment_history_list):
        """
        Initializes the sentiment analysis process.

        Args:
            api_key (str): The Google API key for Gemini.
            sentiment_model_name (str): The name of the Gemini model to use for sentiment analysis.
            audio_queue (multiprocessing.Queue): The queue from which to receive audio chunks.
            sentiment_history_list (multiprocessing.Manager().list): The shared list to store results.
        """
        super().__init__()
        if not genai:
             raise RuntimeError("google-genai library not available. Cannot start SentimentAnalysisProcess.")
        self.api_key = api_key
        self.sentiment_model_name = sentiment_model_name
        self.audio_queue = audio_queue
        self.sentiment_history = sentiment_history_list
        self.exit_flag = multiprocessing.Event()
        self.sentiment_client = None # Initialize client later in run()

    def run(self):
        """Main process function: configures Gemini, listens to the queue, and processes audio or text."""
        print("[Sentiment Process] Starting...")

        # Configure GenAI client within the process
        try:
            # Use the client class directly as in the original script
            self.sentiment_client = genai.Client(api_key=self.api_key)
            # Example: Verify connection by listing models (optional, adds startup time)
            # list(self.sentiment_client.models.list())
            print(f"[Sentiment Process] Configured GenAI client for model: {self.sentiment_model_name}")
        except Exception as e:
            print(f"[Sentiment Process] FATAL ERROR: Configuring GenAI client failed: {e}")
            traceback.print_exc()
            return # Cannot proceed without client

        audio_buffer = []
        last_process_time = time.time()
        chunk_duration_s = 2  # Process audio collected over this duration (reduced from 5s)
        min_chunks_for_analysis = 1 # Minimum number of user chunks to trigger analysis
        max_buffer_chunks = 100 # Reduced safety limit for buffer size

        while not self.exit_flag.is_set():
            collected_new_chunks = False
            # Collect available chunks non-blockingly or with short timeout
            try:
                while len(audio_buffer) < max_buffer_chunks: # Prevent unbounded buffer growth
                    # Use a minimal timeout for maximum responsiveness
                    chunk = self.audio_queue.get(timeout=0.01)

                    # Check if this is a text chunk
                    if 'text' in chunk:
                        # Process text directly with higher priority
                        # Use a higher priority thread for text analysis
                        analysis_thread = threading.Thread(
                            target=self._analyze_text_sentiment_thread,
                            args=(chunk,),
                            daemon=True
                        )
                        # Set thread to higher priority if possible
                        analysis_thread.start()
                        collected_new_chunks = True
                        continue  # Skip adding to audio buffer

                    # Otherwise, treat as audio chunk
                    audio_buffer.append(chunk)
                    collected_new_chunks = True
            except queue.Empty:
                # No more data available right now
                pass
            except Exception as e:
                 print(f"[Sentiment Process] WARNING: Error getting chunk from queue: {e}")
                 # Optional: add a small sleep to prevent tight loop on persistent errors
                 time.sleep(0.1)

            user_chunks_in_buffer = [chunk for chunk in audio_buffer if chunk.get('source') == 'user']
            if not user_chunks_in_buffer:
                 buffer_user_audio_duration_s = 0.0
            else:
                try:
                    buffer_user_audio_duration_s = sum(len(chunk.get('data', b'')) / (chunk.get('rate', 16000) * 2) # 2 bytes/sample for 16-bit
                                                    for chunk in user_chunks_in_buffer)
                except ZeroDivisionError:
                    print("[Sentiment Process] WARNING: Encountered chunk with zero rate.")
                    buffer_user_audio_duration_s = 0.0 # Avoid crash
                except Exception as e:
                    print(f"[Sentiment Process] WARNING: Error calculating buffer duration: {e}")
                    buffer_user_audio_duration_s = 0.0

            current_time = time.time()

            # Decide whether to process based on time elapsed or buffer duration
            should_process = (current_time - last_process_time >= chunk_duration_s and len(user_chunks_in_buffer) >= min_chunks_for_analysis) or \
                             (buffer_user_audio_duration_s >= chunk_duration_s and len(user_chunks_in_buffer) > 0)

            if should_process:
                if user_chunks_in_buffer:
                    # print(f"[Sentiment Process] Triggering analysis for {len(user_chunks_in_buffer)} user chunks ({buffer_user_audio_duration_s:.1f}s)...") # Debug
                    # Process the audio in a separate thread within the process
                    # This prevents blocking the main loop if analysis or upload is slow
                    analysis_thread = threading.Thread(
                        target=self._analyze_audio_sentiment_thread,
                        args=(user_chunks_in_buffer.copy(),), # Pass a copy
                        daemon=True
                    )
                    analysis_thread.start()

                    # Clear processed user chunks from the main buffer
                    audio_buffer = [chunk for chunk in audio_buffer if chunk.get('source') != 'user']
                # else: # Debugging case
                #     print("[Sentiment Process] Processing triggered but no user chunks currently in buffer.")

                last_process_time = current_time # Reset timer after processing attempt

            # If no new chunks were collected and we didn't process, wait efficiently
            elif not collected_new_chunks:
                 # Wait for the exit flag or a timeout, avoids busy-waiting
                 if self.exit_flag.wait(timeout=0.2):
                     break # Exit loop if flag is set during wait

        print("[Sentiment Process] Exiting run loop...")
        # Optional: Add final processing for remaining buffer content here if needed
        # Consider joining any outstanding analysis threads if important

        print("[Sentiment Process] Stopped.")


    def _analyze_text_sentiment_thread(self, text_chunk):
        """
        Internal method performing sentiment analysis on text input.
        Runs in a thread within the SentimentAnalysisProcess.

        Args:
            text_chunk (dict): Dictionary containing 'text', 'source', and 'timestamp'
        """
        if not text_chunk or 'text' not in text_chunk:
            print("[Sentiment Analysis] ERROR: Invalid text chunk received")
            return

        text = text_chunk['text']
        source = text_chunk.get('source', 'unknown')
        timestamp = text_chunk.get('timestamp', time.time())

        try:
            # Ultra-concise prompt for maximum speed
            prompt = """Score these 8 emotions from 0.0-3.0 based on this text: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation. Return ONLY a JSON object like:
{"Joy": 0.7, "Trust": 1.2, "Fear": 0.5, "Surprise": 2.3, "Sadness": 0.1, "Disgust": 0.0, "Anger": 0.3, "Anticipation": 1.7}"""

            # Call the Gemini model
            response = self.sentiment_client.models.generate_content(
                model=self.sentiment_model_name,
                contents=[prompt, text]
            )

            raw_response_text = response.text.strip()
            # Parse the response
            emotion_scores = {
                "Joy": 0, "Trust": 0, "Fear": 0, "Surprise": 0,
                "Sadness": 0, "Disgust": 0, "Anger": 0, "Anticipation": 0
            }

            # Clean and parse the JSON response
            cleaned_text = raw_response_text
            # Remove any markdown code block markers
            if cleaned_text.startswith('```') and cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[3:-3].strip()
            # Remove any language identifier after the first code block marker
            if cleaned_text.startswith('```'):
                cleaned_text = '```'.join(cleaned_text.split('```')[1:]).strip()
            # Remove 'json' if it appears at the start
            if cleaned_text.startswith('json'):
                cleaned_text = cleaned_text[4:].strip()

            try:
                # Attempt to parse the cleaned JSON response
                parsed_json = json.loads(cleaned_text)
                if isinstance(parsed_json, dict):
                    # Validate and update scores, keeping defaults for missing/invalid keys
                    # Support both integer and float values for more continuous emotional representation
                    for emotion in emotion_scores.keys():
                        if emotion in parsed_json and isinstance(parsed_json[emotion], (int, float)) and 0 <= parsed_json[emotion] <= 3:
                            # Convert to float for continuous representation
                            emotion_scores[emotion] = float(parsed_json[emotion])
                else:
                    print(f"[Sentiment Analysis] WARNING: Parsed JSON is not a dictionary: '{cleaned_text}'. Using default scores.")
            except json.JSONDecodeError:
                # Handle cases where the response is not valid JSON
                if cleaned_text: # Only warn if the response wasn't empty
                    print(f"[Sentiment Analysis] WARNING: Failed to decode JSON response: '{cleaned_text}'. Using default scores.")

            # Store the result
            result = {
                "timestamp": timestamp,
                "emotion_scores": emotion_scores,
                "raw_response": raw_response_text,
                "source": source,
                "text": text[:100] + "..." if len(text) > 100 else text  # Store truncated text for reference
            }
            self.sentiment_history.append(result)

            # Minimal logging to improve performance
            pass

        except Exception as e:
            print(f"[Sentiment Analysis] ERROR analyzing text: {e}")
            error_result = {
                "timestamp": timestamp,
                "error": str(e),
                "source": source,
                "traceback": traceback.format_exc()
            }
            self.sentiment_history.append(error_result)

    def _analyze_audio_sentiment_thread(self, user_chunks):
        """
        Internal method performing the sentiment analysis (upload, API call, result parsing).
        Runs in a thread within the SentimentAnalysisProcess.
        """
        if not user_chunks:
            return

        # Combine user audio chunks and get metadata
        try:
            combined_audio = b''.join([chunk['data'] for chunk in user_chunks])
            # Assume consistency in sample rate for user audio chunks
            sample_rate = user_chunks[0].get('rate', 16000)
            channels = 1 # Assume mono
            sampwidth = 2 # Assume 16-bit PCM

            # Avoid processing very short/empty audio segments
            min_duration_s = 0.5
            actual_duration_s = 0.0 # Default
            if sample_rate > 0 and sampwidth > 0 and channels > 0:
                actual_duration_s = len(combined_audio) / (sample_rate * sampwidth * channels)

            if actual_duration_s < min_duration_s:
                 # print(f"[Sentiment Analysis] Skipping analysis for very short audio segment ({actual_duration_s:.2f}s < {min_duration_s}s).")
                 return

        except Exception as e:
             print(f"[Sentiment Analysis] ERROR: Failed to combine audio chunks: {e}")
             traceback.print_exc()
             return

        temp_file_path = None
        audio_file_resource = None # To store the uploaded file object from Gemini API

        try:
            # Create a temporary WAV file using a context manager for safety
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_f:
                temp_file_path = temp_f.name
                # Write the combined audio data to the WAV file
                with wave.open(temp_f, 'wb') as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(sample_rate)
                    wf.writeframes(combined_audio)

            # Upload the temporary file to Gemini
            # print(f"[Sentiment Analysis] Uploading {actual_duration_s:.1f}s audio from temp file: {temp_file_path}") # Debug
            audio_file_resource = self.sentiment_client.files.upload(file=temp_file_path)
            # print(f"[Sentiment Analysis] Uploaded file. URI: {audio_file_resource.uri}, State: {audio_file_resource.state}") # Debug

            # --- IMPORTANT: Wait for file processing ---
            # Gemini needs time to process the uploaded file before it can be used.
            # Poll the file status until it's 'ACTIVE'.
            polling_start_time = time.time()
            max_polling_wait_s = 60 # Max time to wait for processing
            while audio_file_resource.state.name == "PROCESSING":
                if time.time() - polling_start_time > max_polling_wait_s:
                     print(f"[Sentiment Analysis] ERROR: File processing timed out after {max_polling_wait_s}s. URI: {audio_file_resource.uri}")
                     raise TimeoutError("Gemini file processing timed out")

                # print(f"[Sentiment Analysis] Waiting for file processing (current state: {audio_file_resource.state.name})...") # Debug
                time.sleep(2) # Check every 2 seconds (adjust polling interval as needed)
                try:
                    audio_file_resource = self.sentiment_client.files.get(name=audio_file_resource.name)
                except Exception as get_err:
                    print(f"[Sentiment Analysis] ERROR: Failed to get file status during polling: {get_err}")
                    # Decide how to handle: retry, abort, etc. Aborting for now.
                    raise  # Re-raise to be caught by the outer try-except

            if audio_file_resource.state.name != "ACTIVE":
                print(f"[Sentiment Analysis] ERROR: Uploaded file failed processing. Final state: {audio_file_resource.state.name}. URI: {audio_file_resource.uri}")
                # Analysis cannot proceed
                # Clean up before returning
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.unlink(temp_file_path)
                    except Exception as e: print(f"[Sentiment Analysis] WARNING: Failed to delete temp file after failed processing: {e}")
                # Attempt to delete the failed file from Gemini storage
                try: self.sentiment_client.files.delete(name=audio_file_resource.name)
                except Exception as e: print(f"[Sentiment Analysis] WARNING: Failed to delete failed Gemini file {audio_file_resource.name}: {e}")
                return

            # print(f"[Sentiment Analysis] File is ACTIVE. URI: {audio_file_resource.uri}") # Debug

            # Ultra-concise prompt for maximum speed
            prompt = """Score these 8 emotions from 0.0-3.0 based on this audio: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation. Return ONLY a JSON object like:
{"Joy": 0.7, "Trust": 1.2, "Fear": 0.5, "Surprise": 2.3, "Sadness": 0.1, "Disgust": 0.0, "Anger": 0.3, "Anticipation": 1.7}"""

            # Call the Gemini model via the client's model interface
            # print("[Sentiment Analysis] Calling Gemini API for sentiment...") # Debug
            response = self.sentiment_client.models.generate_content(
                model=self.sentiment_model_name,
                contents=[prompt, audio_file_resource], # Pass the file resource directly
                # generation_config=genai.types.GenerationConfig(
                #     candidate_count=1,
                #     temperature=0.1 # Low temp for classification
                #     )
            )
            if not response.candidates:
                 sentiment_text = "" # Handle gracefully
            else:
                # Assuming the first candidate contains the result
                sentiment_text = response.text.strip()

            # --- Result Parsing (New: Expecting JSON with 8 emotion scores) ---
            emotion_scores = { # Default scores
                "Joy": 0, "Trust": 0, "Fear": 0, "Surprise": 0,
                "Sadness": 0, "Disgust": 0, "Anger": 0, "Anticipation": 0
            }
            raw_response_text = sentiment_text # Store the raw text for debugging

            # Clean the response: Remove markdown fences and extra whitespace
            cleaned_text = sentiment_text
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:] # Remove ```json
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3] # Remove ```
            cleaned_text = cleaned_text.strip() # Remove surrounding whitespace

            try:
                # Attempt to parse the cleaned JSON response
                parsed_json = json.loads(cleaned_text)
                if isinstance(parsed_json, dict):
                     # Validate and update scores, keeping defaults for missing/invalid keys
                     # Support both integer and float values for more continuous emotional representation
                     for emotion in emotion_scores.keys():
                         if emotion in parsed_json and isinstance(parsed_json[emotion], (int, float)) and 0 <= parsed_json[emotion] <= 3:
                             # Convert to float for continuous representation
                             emotion_scores[emotion] = float(parsed_json[emotion])
                         # else: # Optional: Log if an expected emotion is missing or invalid
                         #     print(f"[Sentiment Analysis] WARNING: Missing or invalid score for '{emotion}' in response: {parsed_json.get(emotion)}")
                else:
                     print(f"[Sentiment Analysis] WARNING: Parsed JSON is not a dictionary: '{cleaned_text}'. Using default scores.")
            except json.JSONDecodeError:
                 # Handle cases where the response is not valid JSON
                 if cleaned_text: # Only warn if the response wasn't empty
                     print(f"[Sentiment Analysis] WARNING: Failed to decode JSON response: '{cleaned_text}'. Using default scores.")
            # --- End New Result Parsing ---

            # Store the result in the shared list (passed during initialization)
            # Get the source from the first user chunk if available
            source = "user"  # Default to user
            if user_chunks and isinstance(user_chunks[0], dict) and "source" in user_chunks[0]:
                source = user_chunks[0]["source"]

            result = {
                "timestamp": time.time(),
                "emotion_scores": emotion_scores, # Store the dictionary of scores
                "raw_response": raw_response_text,
                "audio_duration_s": actual_duration_s,
                "source": source # Include the source
            }
            self.sentiment_history.append(result)

            # No logging for better performance

        except TimeoutError as e:
            # Handle specific timeout error from polling loop
             print(f"[Sentiment Analysis] ERROR: {e}")
             # Optional: Record error
             self.sentiment_history.append({"timestamp": time.time(), "error": str(e)})
        except Exception as e:
            print(f"[Sentiment Analysis] ERROR during analysis thread: {e}")
            traceback.print_exc()
            # Optional: Record the error in the history list
            self.sentiment_history.append({"timestamp": time.time(), "error": str(e), "traceback": traceback.format_exc()})

        finally:
            # --- Cleanup ---
            # Delete the file from the Gemini service to avoid clutter/costs
            if audio_file_resource:
                try:
                    # print(f"[Sentiment Analysis] Deleting uploaded Gemini file: {audio_file_resource.name}") # Debug
                    self.sentiment_client.files.delete(name=audio_file_resource.name)
                except Exception as delete_err:
                    # Log warning, but don't prevent other cleanup
                    print(f"[Sentiment Analysis] WARNING: Failed to delete uploaded Gemini file {audio_file_resource.name}: {delete_err}")

            # Delete the local temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    # print(f"[Sentiment Analysis] Deleting local temp file: {temp_file_path}") # Debug
                    os.unlink(temp_file_path)
                except Exception as e:
                    print(f"[Sentiment Analysis] WARNING: Could not delete local temporary file {temp_file_path}: {e}")

    def stop(self):
        """Signals the process to stop listening and exit gracefully."""
        print("[Sentiment Process] Received stop signal.")
        self.exit_flag.set()

class SentimentAnalysisManager:
    """
    Manager class that provides a clean interface between the realtime conversation process
    and the sentiment analysis process. This class encapsulates all the multiprocessing
    details and provides a simple API for sending audio data and retrieving sentiment results.
    """

    def __init__(self, google_api_key, sentiment_model_name="models/gemini-2.0-flash"):
        """
        Initialize the sentiment analysis manager.

        Args:
            google_api_key (str): The Google API key for Gemini.
            sentiment_model_name (str): The name of the Gemini model to use for sentiment analysis.
        """
        self.google_api_key = google_api_key
        self.sentiment_model_name = sentiment_model_name
        self.manager = None
        self.audio_queue = None
        self.sentiment_history_list = None
        self.sentiment_process = None
        self.shared_emotion_scores = None
        self.sentiment_update_thread = None
        self.sentiment_stop_event = threading.Event()
        self.sentiment_file_path = None

        # Flag to control which source to use for sentiment analysis
        # Default to text-only mode for better performance
        self.use_text_only = True

        # Initialize components during creation
        self.is_initialized = False
        self.is_initialized = self._initialize_components()

    def is_running(self):
        """Check if the sentiment analysis process is running.

        Returns:
            bool: True if the process is running, False otherwise
        """
        return (self.sentiment_process is not None and
                self.sentiment_process.is_alive() and
                self.sentiment_update_thread is not None and
                self.sentiment_update_thread.is_alive())

    def _initialize_components(self):
        """
        Initialize the multiprocessing components needed for sentiment analysis.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        # Clean up any existing components first
        self._cleanup_components()

        try:
            if not self.google_api_key:
                logger.warning("Sentiment analysis manager initialized without API key.")
                return False

            if not genai:
                logger.warning("Sentiment analysis manager initialized without genai library.")
                return False

            # Test the Google API key by creating a client
            try:
                # Just creating the client is enough to verify the API key
                _ = genai.Client(api_key=self.google_api_key)
                logger.info(f"Successfully verified Google API key for model {self.sentiment_model_name}.")
            except Exception as e:
                logger.error(f"Failed to initialize Google API client: {e}")
                return False

            # Initialize multiprocessing components
            try:
                self.manager = multiprocessing.Manager()
                self.audio_queue = self.manager.Queue(maxsize=500)  # Queue for audio chunks
                self.sentiment_history_list = self.manager.list()  # Shared list for results

                # Initialize the shared emotion state with a multiprocessing shared dictionary
                shared_dict = self.manager.dict({
                    'user': {emotion: 0 for emotion in CORE_EMOTIONS},
                    'assistant': {emotion: 0 for emotion in CORE_EMOTIONS},
                    'last_update_time': time.time()
                })
                self.shared_emotion_scores = SharedEmotionState(shared_dict)

                logger.info(f"Sentiment analysis components initialized for model {self.sentiment_model_name}.")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize multiprocessing components: {e}")
                self._cleanup_components()  # Clean up partial initialization
                return False
        except Exception as e:
            logger.error(f"Failed to initialize multiprocessing components for sentiment analysis: {e}")
            # Clean up partial initialization
            self._cleanup_components()
            return False

    def _cleanup_components(self):
        """
        Clean up multiprocessing components.
        """
        if self.manager:
            try:
                self.manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down multiprocessing manager: {e}")
            self.manager = None
            self.audio_queue = None
            self.sentiment_history_list = None
            self.shared_emotion_scores = None

    def start(self, sentiment_file_path=None):
        """
        Start the sentiment analysis process.

        Args:
            sentiment_file_path (str, optional): Path to save sentiment results.

        Returns:
            bool: True if started successfully, False otherwise.
        """
        if self.sentiment_process is not None and self.sentiment_process.is_alive():
            logger.info("Sentiment analysis process already running.")
            return True

        # Check if initialization was successful
        if not self.is_initialized:
            logger.error("Cannot start sentiment analysis process: initialization failed.")
            # Try to initialize again
            self.is_initialized = self._initialize_components()
            if not self.is_initialized:
                return False

        # Double-check that components are available
        # Note: sentiment_history_list is a list, which is truthy even when empty
        if self.audio_queue is None or self.sentiment_history_list is None or self.shared_emotion_scores is None:
            logger.error("Cannot start sentiment analysis process: components not initialized properly.")
            logger.error(f"audio_queue: {self.audio_queue}, sentiment_history_list: {self.sentiment_history_list}, shared_emotion_scores: {self.shared_emotion_scores}")
            return False

        self.sentiment_file_path = sentiment_file_path

        try:
            # Create and start the sentiment analysis process
            self.sentiment_process = SentimentAnalysisProcess(
                api_key=self.google_api_key,
                sentiment_model_name=self.sentiment_model_name,
                audio_queue=self.audio_queue,
                sentiment_history_list=self.sentiment_history_list
            )
            self.sentiment_process.start()
            logger.info("Sentiment analysis process started successfully.")

            # Start the sentiment update thread
            self.sentiment_stop_event.clear()
            self.sentiment_update_thread = threading.Thread(
                target=self._continuously_update_sentiment,
                daemon=True
            )
            self.sentiment_update_thread.start()
            logger.info("Sentiment update thread started.")

            return True
        except Exception as e:
            logger.error(f"Failed to start sentiment analysis process: {e}")
            if self.sentiment_process:
                try:
                    self.sentiment_process.terminate()
                except Exception:
                    pass
                self.sentiment_process = None
            return False

    def save_results(self, custom_file_path=None):
        """
        Explicitly save the current sentiment results to a file.

        Args:
            custom_file_path (str, optional): Custom file path to save results to.
                If not provided, uses the path set during start().

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        if not self.sentiment_history_list:
            logger.warning("No sentiment results to save.")
            return False

        file_path = custom_file_path or self.sentiment_file_path
        if not file_path:
            logger.warning("No file path provided for saving sentiment results.")
            return False

        try:
            # Convert the shared list to a regular list
            sentiment_results = list(self.sentiment_history_list)

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(sentiment_results, f, indent=2)

            logger.info(f"Saved {len(sentiment_results)} sentiment results to {file_path}")
            print(f"Saved sentiment analysis results to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving sentiment results: {e}")
            return False

    def stop(self, custom_file_path=None):
        """
        Stop the sentiment analysis process.

        Args:
            custom_file_path (str, optional): Custom file path to save results to before stopping.
                If not provided, uses the path set during start().

        Returns:
            bool: True if stopped successfully, False otherwise.
        """
        # Save results one last time before stopping
        if custom_file_path:
            self.save_results(custom_file_path=custom_file_path)
        elif self.sentiment_file_path:
            self.save_results()

        # Stop the update thread first
        if self.sentiment_update_thread and self.sentiment_update_thread.is_alive():
            self.sentiment_stop_event.set()
            self.sentiment_update_thread.join(timeout=5)
            if self.sentiment_update_thread.is_alive():
                logger.warning("Sentiment update thread did not terminate gracefully.")
            self.sentiment_update_thread = None

        # Stop the sentiment process
        if self.sentiment_process and self.sentiment_process.is_alive():
            try:
                self.sentiment_process.stop()
                self.sentiment_process.join(timeout=15)  # Wait longer for potential network ops

                if self.sentiment_process.is_alive():
                    logger.warning("Sentiment process did not terminate gracefully. Forcing termination.")
                    self.sentiment_process.terminate()
                    self.sentiment_process.join()
            except Exception as e:
                logger.error(f"Error stopping sentiment process: {e}")
                return False

        self.sentiment_process = None
        return True

    def send_audio_chunk(self, audio_data, sample_rate, source="user"):
        """
        Send an audio chunk for sentiment analysis.

        Args:
            audio_data (bytes): The audio data to analyze.
            sample_rate (int): The sample rate of the audio data.
            source (str): The source of the audio ("user" or "model").

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.is_running() or not self.audio_queue:
            return False

        # Skip audio processing if we're in text-only mode
        if self.use_text_only:
            return True  # Return True to indicate success (we're intentionally skipping)

        try:
            # If this is a user audio chunk, check if we should skip it
            if source == "user" and self.shared_emotion_scores:
                # Skip if assistant is responding
                if self.shared_emotion_scores.is_assistant_responding():
                    # Skip user audio chunks while assistant is responding to prevent anomalous updates
                    logger.debug("Skipping user audio chunk while assistant is responding")
                    return True  # Return True to indicate success (we're intentionally skipping)

                # Skip if VAD has not detected speech
                if not self.shared_emotion_scores.is_vad_speech_detected():
                    # Skip user audio chunks when VAD has not detected speech to prevent spurious updates
                    logger.debug("Skipping user audio chunk when VAD has not detected speech")
                    return True  # Return True to indicate success (we're intentionally skipping)

                # For silence, we'll still process the audio but log it
                # This ensures the visualization continues to update
                if self.shared_emotion_scores.is_silence_detected():
                    logger.debug("Processing user audio chunk during silence period (for visualization)")

            audio_chunk = {
                "source": source,
                "data": audio_data,
                "rate": sample_rate,
                "timestamp": time.time()
            }
            # Use put_nowait to avoid blocking the main thread
            # If the queue is full, we'll just skip this chunk
            try:
                self.audio_queue.put_nowait(audio_chunk)
                return True
            except queue.Full:
                logger.warning("Sentiment analysis audio queue is full. Skipping chunk.")
                return False
        except Exception as e:
            logger.error(f"Error sending audio chunk to sentiment queue: {e}")
            return False

    def send_text(self, text, source="user"):
        """
        Send text for sentiment analysis.

        Args:
            text (str): The text to analyze.
            source (str): The source of the text ("user" or "model").

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.is_running() or not self.audio_queue:
            return False

        # Skip text processing if we're not in text-only mode
        if not self.use_text_only:
            return True  # Return True to indicate success (we're intentionally skipping)

        try:
            # Ultra-efficient version - minimal checks
            if source == "user" and self.shared_emotion_scores:
                # In text chat mode, always set VAD to true
                if not self.shared_emotion_scores.is_audio_mode():
                    self.shared_emotion_scores.set_vad_speech_detected(True)

            # Skip very short text
            if len(text.strip()) < 5:
                return True

            # Create and send the request
            text_sentiment_request = {
                "source": source,
                "text": text,
                "timestamp": time.time()
            }

            try:
                self.audio_queue.put_nowait(text_sentiment_request)
                return True
            except:
                return False
        except:
            return False

    def get_current_emotion_scores(self):
        """
        Get the current emotion scores.

        Returns:
            dict: A dictionary of emotion scores, or None if not available.
        """
        if not self.shared_emotion_scores:
            return None

        try:
            # Ultra-efficient version
            user_scores, assistant_scores = self.shared_emotion_scores.get_emotion_scores()
            return {'user': user_scores, 'assistant': assistant_scores}
        except:
            return None

    def is_running(self):
        """
        Check if the sentiment analysis process is running.

        Returns:
            bool: True if running, False otherwise.
        """
        return self.sentiment_process is not None and self.sentiment_process.is_alive()

    def _continuously_update_sentiment(self):
        """
        Continuously update the shared emotion scores from the sentiment history list.
        This runs in a separate thread.
        """
        last_processed_index = -1

        while not self.sentiment_stop_event.is_set():
            try:
                # Check if there are new results to process
                current_len = len(self.sentiment_history_list)
                if current_len > last_processed_index + 1:
                    # Process new results - only the most recent one for maximum efficiency
                    result = self.sentiment_history_list[current_len - 1]

                    # Update the appropriate shared dictionary based on source
                    if 'emotion_scores' in result and 'source' in result:
                        source = result.get('source', 'user')

                        # Ultra-efficient update - no checks, just update
                        if source == 'user':
                            self.shared_emotion_scores.update_emotion_scores('user', result['emotion_scores'])
                        elif source == 'model' or source == 'assistant':
                            self.shared_emotion_scores.update_emotion_scores('assistant', result['emotion_scores'])

                    # Update the index to the current length
                    last_processed_index = current_len - 1

                # Sleep for a minimal duration
                time.sleep(0.01)  # Check 100 times per second
            except:
                # No logging, just continue
                time.sleep(0.1)


# Example usage (for testing purposes, would be integrated elsewhere)
if __name__ == '__main__':
    print("Running sentiment_analyzer.py directly for testing...")

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    try:
        # Load API key securely (replace with your actual key loading)
        # Assumes .api_key.json is in the PARENT directory relative to gpt_realtime/
        api_key_path = os.path.join(os.path.dirname(__file__), '..', '.api_key.json')
        with open(api_key_path, "r") as f:
             api_keys = json.load(f)
             google_api_key = api_keys.get("gemini_api_key") # Use the correct key name from gemini_live_audio.py
        if not google_api_key:
            raise ValueError("Gemini API key ('gemini_api_key') not found")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"FATAL ERROR loading API key from {api_key_path}: {e}")
        print("Please ensure '.api_key.json' exists in the project root directory and contains 'gemini_api_key'.")
        exit(1)
    if not genai:
        print("FATAL ERROR: google-genai library not loaded.")
        exit(1)

    # Use the same model name as in gemini_live_audio.py
    SENTIMENT_MODEL = "models/gemini-2.0-flash"

    print(f"Using sentiment model: {SENTIMENT_MODEL}")

    # Create the sentiment analysis manager
    sentiment_manager = SentimentAnalysisManager(
        google_api_key=google_api_key,
        sentiment_model_name=SENTIMENT_MODEL
    )

    # Start the sentiment analysis process
    if not sentiment_manager.start():
        print("Failed to start sentiment analysis process.")
        exit(1)

    # --- Simulate sending audio chunks ---
    print("\nSimulating sending audio chunks (using silence)...")
    SAMPLE_RATE = 16000
    CHUNK_DURATION_S = 0.5 # Duration of each simulated chunk
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)
    SILENCE_CHUNK = b'\x00\x00' * CHUNK_SAMPLES # 16-bit silence

    # Simulate some user speech (12 seconds worth)
    print("Simulating user speech...")
    num_user_chunks = int(12 / CHUNK_DURATION_S)
    for i in range(num_user_chunks):
        # In a real scenario, 'data' would be actual audio bytes
        chunk_data = SILENCE_CHUNK # Replace with actual audio data for real testing
        sentiment_manager.send_audio_chunk(chunk_data, SAMPLE_RATE, "user")
        time.sleep(0.05) # Faster simulation

    # Simulate some model speech (5 seconds worth - should be ignored)
    print("Simulating model speech (should be ignored)...")
    num_model_chunks = int(5 / CHUNK_DURATION_S)
    for i in range(num_model_chunks):
        sentiment_manager.send_audio_chunk(SILENCE_CHUNK, 24000, "model")
        time.sleep(0.05)

    # Simulate sending text
    print("Simulating sending text...")
    sentiment_manager.send_text("I'm feeling really happy today!", "user")
    sentiment_manager.send_text("That's great to hear. I'm glad you're doing well.", "model")

    # Wait for processing to occur
    wait_time = 15
    print(f"\nWaiting {wait_time}s for sentiment analysis to process...")
    time.sleep(wait_time)

    # Get current emotion scores
    print("\nCurrent emotion scores:")
    scores = sentiment_manager.get_current_emotion_scores()
    if scores:
        scores_str = ", ".join([f"{k}: {v}" for k, v in scores.items()])
        print(f"Scores: [{scores_str}]")
    else:
        print("No emotion scores available.")

    # Stop the sentiment analysis process
    print("\nStopping sentiment analysis process...")
    sentiment_manager.stop()

    print("\nTest finished.")