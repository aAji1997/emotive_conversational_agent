import multiprocessing
import queue
import threading
import time
import traceback
import json
import tempfile
import wave
import os
try:
    from google import genai
except ImportError:
    print("ERROR: The 'google-genai' library is required for sentiment analysis.")
    print("Please install it using: pip install google-genai")
    
    raise ImportError("Missing dependency: google-genai")


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
        chunk_duration_s = 10  # Process audio collected over this duration
        min_chunks_for_analysis = 2 # Minimum number of user chunks to trigger analysis
        max_buffer_chunks = 500 # Safety limit for buffer size

        while not self.exit_flag.is_set():
            collected_new_chunks = False
            # Collect available chunks non-blockingly or with short timeout
            try:
                while len(audio_buffer) < max_buffer_chunks: # Prevent unbounded buffer growth
                    # Use timeout to allow checking exit flag periodically
                    chunk = self.audio_queue.get(timeout=0.1)

                    # Check if this is a text chunk
                    if 'text' in chunk:
                        # Process text directly
                        analysis_thread = threading.Thread(
                            target=self._analyze_text_sentiment_thread,
                            args=(chunk,),
                            daemon=True
                        )
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
            # Define the prompt for text sentiment analysis
            prompt = """Analyze the emotional state expressed in this text based on Plutchik's wheel of emotions. Focus on the eight core emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation.

For each core emotion, provide a score from 0 to 3 indicating its intensity in the text:
- 0: Emotion is absent.
- 1: Low intensity (e.g., Serenity for Joy, Apprehension for Fear).
- 2: Medium intensity (e.g., Joy for Joy, Fear for Fear).
- 3: High intensity (e.g., Ecstasy for Joy, Terror for Fear).

Return ONLY a valid JSON object mapping each of the 8 core emotion strings to its integer score (0, 1, 2, or 3). Example format:
{"Joy": 0, "Trust": 1, "Fear": 0, "Surprise": 2, "Sadness": 0, "Disgust": 0, "Anger": 0, "Anticipation": 1}"""

            # Call the Gemini model
            response = self.sentiment_client.models.generate_content(
                model=self.sentiment_model_name,
                contents=[prompt, text]
            )

            raw_response_text = response.text.strip()
            print(f"[Sentiment Analysis] Raw text response: '{raw_response_text}'")

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
                    for emotion in emotion_scores.keys():
                        if emotion in parsed_json and isinstance(parsed_json[emotion], int) and 0 <= parsed_json[emotion] <= 3:
                            emotion_scores[emotion] = parsed_json[emotion]
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

            print(f"[Sentiment Analysis] Text Result Scores for {source}: {emotion_scores}")

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

            # Define the prompt precisely matching the one in gemini_live_audio.py
            prompt = """Analyze the user's emotional state expressed in this audio based on Plutchik's wheel of emotions. Focus on the eight core emotions: Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation.

For each core emotion, provide a score from 0 to 3 indicating its intensity in the audio:
- 0: Emotion is absent.
- 1: Low intensity (e.g., Serenity for Joy, Apprehension for Fear).
- 2: Medium intensity (e.g., Joy for Joy, Fear for Fear).
- 3: High intensity (e.g., Ecstasy for Joy, Terror for Fear).

Return ONLY a valid JSON object mapping each of the 8 core emotion strings to its integer score (0, 1, 2, or 3). Example format:
{"Joy": 0, "Trust": 1, "Fear": 0, "Surprise": 2, "Sadness": 0, "Disgust": 0, "Anger": 0, "Anticipation": 1}"""

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
            print(f"[Sentiment Analysis] Raw response text: '{response.text}'") # Debug

            if not response.candidates:
                 print("[Sentiment Analysis] WARNING: Received no candidates in response from Gemini.")
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
                     for emotion in emotion_scores.keys():
                         if emotion in parsed_json and isinstance(parsed_json[emotion], int) and 0 <= parsed_json[emotion] <= 3:
                             emotion_scores[emotion] = parsed_json[emotion]
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
            result = {
                "timestamp": time.time(),
                "emotion_scores": emotion_scores, # Store the dictionary of scores
                "raw_response": raw_response_text,
                "audio_duration_s": actual_duration_s
            }
            self.sentiment_history.append(result)

            print(f"[Sentiment Analysis] Result Scores: {emotion_scores} for {actual_duration_s:.1f}s audio segment.") # Debug

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

# Example usage (for testing purposes, would be integrated elsewhere)
if __name__ == '__main__':
    print("Running sentiment_analyzer.py directly for testing...")

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


    # --- Setup multiprocessing components ---
    manager = multiprocessing.Manager()
    audio_q = manager.Queue(maxsize=200) # Increased queue size for simulation
    results_list = manager.list()

    # --- Start the sentiment process ---
    sentiment_proc = None
    try:
        sentiment_proc = SentimentAnalysisProcess(
            api_key=google_api_key,
            sentiment_model_name=SENTIMENT_MODEL,
            audio_queue=audio_q,
            sentiment_history_list=results_list
        )
        sentiment_proc.start()
    except RuntimeError as e:
        print(f"FATAL ERROR starting sentiment process: {e}")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR during sentiment process startup: {e}")
        traceback.print_exc()
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
        audio_chunk = {
            'source': 'user',
            'data': chunk_data,
            'rate': SAMPLE_RATE,
            'timestamp': time.time()
        }
        try:
             audio_q.put(audio_chunk, timeout=1)
        except queue.Full:
             print("Test Warning: Audio queue full during user simulation.")
             break
        # time.sleep(CHUNK_DURATION_S * 0.8) # Simulate real-time arrival slightly faster
        time.sleep(0.05) # Faster simulation

    # Simulate some model speech (5 seconds worth - should be ignored)
    print("Simulating model speech (should be ignored)...")
    num_model_chunks = int(5 / CHUNK_DURATION_S)
    for i in range(num_model_chunks):
        audio_chunk = {
            'source': 'model',
            'data': SILENCE_CHUNK,
            'rate': 24000, # Example different rate for model
            'timestamp': time.time()
        }
        try:
            audio_q.put(audio_chunk, timeout=1)
        except queue.Full:
            print("Test Warning: Audio queue full during model simulation.")
            break
        # time.sleep(CHUNK_DURATION_S * 0.8)
        time.sleep(0.05)

    # Wait for processing to occur (adjust time based on chunk_duration_s in SentimentAnalysisProcess)
    # It processes in 10s batches, so wait at least that long + buffer time + API time
    wait_time = 15
    print(f"\nWaiting {wait_time}s for sentiment analysis to process...")
    time.sleep(wait_time)

    # --- Stop the process and collect results ---
    print("\nStopping sentiment process...")
    if sentiment_proc and sentiment_proc.is_alive():
        sentiment_proc.stop()
        sentiment_proc.join(timeout=15) # Wait longer for potential network ops

        if sentiment_proc.is_alive():
            print("WARNING: Sentiment process did not terminate gracefully. Forcing termination.")
            sentiment_proc.terminate()
            sentiment_proc.join()
    elif sentiment_proc:
         print("Sentiment process already terminated.")
    else:
         print("Sentiment process was not started.")

    print("\n--- Sentiment Analysis Results ---")
    # Convert shared list to regular list for easier handling
    final_results = list(results_list)

    if not final_results:
        print("No sentiment results were generated (or process failed). Check logs above.")
    else:
        for result in final_results:
            if 'error' in result:
                 ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                 print(f"[{ts}] ERROR: {result['error']}")
                 if 'traceback' in result:
                      print(result['traceback'])
            else:
                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.get('timestamp', time.time())))
                duration = result.get('audio_duration_s', 'N/A')
                scores = result.get('emotion_scores', {}) # Get the scores dictionary
                raw = result.get('raw_response', '')
                # Format the scores for printing
                scores_str = ", ".join([f"{k}: {v}" for k, v in scores.items()])
                print(f"[{ts}] Scores: [{scores_str}], Duration: {duration:.1f}s, Raw: '{raw}'")

    print("\nTest finished.")