import pyaudio
import threading
import queue
import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioHandler:
    """Handles audio input and output using PyAudio."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = b''
        self.chunk_size = 1024  # Number of audio frames per buffer
        self.format = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.channels = 1  # Mono audio
        self.rate = 24000  # Sampling rate in Hz
        self.is_recording = False

        # Lock for audio playback to avoid overlapping audio
        self.playback_lock = threading.Lock()
        self.currently_playing = False

        # ===== MEMORY QUERY SUPPRESSION MECHANISM - AUDIO HANDLING =====
        # This is part of the bespoke solution to fix the "I don't know/Actually I do know" problem
        # When a memory-related query is detected, we need to suppress the initial incorrect response
        # in the audio playback.
        #
        # The approach works as follows:
        # 1. Set memory_retrieval_active flag when a memory query is detected
        # 2. Track which response we're on using memory_response_count
        # 3. In the play_audio method, we check these flags and only play the second response
        #
        # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
        # It's critical for providing a good user experience with memory-related queries.
        self.memory_retrieval_active = False
        self.memory_response_count = 0

        # List available audio devices
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)

    def start_audio_stream(self):
        """Start the audio input stream."""
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            self.stream = None

    def stop_audio_stream(self):
        """Stop the audio input stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            self.stop_audio_stream()
        self.p.terminate()

    def start_recording(self):
        """Start continuous recording."""
        self.is_recording = True
        self.audio_buffer = b''
        self.start_audio_stream()

    def stop_recording(self):
        """Stop recording and return the recorded audio."""
        self.is_recording = False
        self.stop_audio_stream()
        return self.audio_buffer

    def record_chunk(self):
        """Record a single chunk of audio."""
        if self.stream and self.is_recording:
            try:
                data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )
                self.audio_buffer += data
                return data
            except Exception as e:
                logger.error(f"Error reading audio chunk: {e}")
                return None
        return None

    def set_memory_retrieval_mode(self, active):
        """Set whether we're in memory retrieval mode.

        This is part of the memory query suppression mechanism that prevents the system
        from playing the initial incorrect response for memory-related queries.

        Args:
            active (bool): Whether to activate memory retrieval mode
        """
        with self.playback_lock:
            # If we're entering memory retrieval mode, reset the response counter
            if active and not self.memory_retrieval_active:
                self.memory_response_count = 0
            self.memory_retrieval_active = active
            logger.info(f"Memory retrieval mode set to: {active}")

    def play_audio(self, audio_data):
        """Play audio data."""
        # Skip if there's no audio data
        if not audio_data or len(audio_data) == 0:
            logger.warning("No audio data provided to play")
            return

        # Check if already playing
        if self.currently_playing:
            logger.warning("Already playing audio, cancelling new playback")
            return

        # ===== MEMORY QUERY SUPPRESSION IMPLEMENTATION =====
        # This is where we actually suppress the incorrect response for memory-related queries.
        # When in memory retrieval mode, we track which response we're on and skip the first one.
        #
        # The first response would say something like "I don't know where you live" which is incorrect.
        # The second response (after memory context is added) will have the correct information.
        #
        # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
        # It's critical for providing a good user experience with memory-related queries.
        if self.memory_retrieval_active:
            with self.playback_lock:
                self.memory_response_count += 1

                # Skip the first response (which is the incorrect one)
                if self.memory_response_count == 1:
                    logger.info("Skipping first audio response during memory retrieval")
                    return

        def play():
            # Set a flag to indicate we're playing audio
            with self.playback_lock:
                self.currently_playing = True

            try:
                stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    output=True
                )
                # Use a slightly larger buffer for writing to avoid buffer underruns
                buffer_size = 1024 * 4
                for i in range(0, len(audio_data), buffer_size):
                    chunk = audio_data[i:i + buffer_size]
                    stream.write(chunk)
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
            finally:
                # Clear the playing flag when done
                with self.playback_lock:
                    self.currently_playing = False

            # logger.debug("Audio playback complete")

        # Use daemon=True to ensure thread doesn't prevent program exit
        playback_thread = threading.Thread(target=play, daemon=True)
        playback_thread.start()

    def stop_playback(self):
        """Stop any ongoing audio playback."""
        # Just set the flag to false - won't affect currently playing audio
        # but will prevent new playback from starting
        with self.playback_lock:
            if self.currently_playing:
                logger.info("Marking playback as stopped")
            self.currently_playing = False

    def emergency_stop(self):
        """Forcefully terminate all audio processes immediately."""
        logger.info("EMERGENCY STOP: Forcefully terminating all audio processes")

        # Stop recording
        self.is_recording = False

        # Stop playback
        with self.playback_lock:
            self.currently_playing = False

        # Force close any open streams
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                logger.error(f"Error closing audio stream during emergency stop: {e}")

        # Clear audio buffer
        self.audio_buffer = b''


class TranscriptProcessor:
    """Processes transcripts in a separate thread to avoid blocking the main conversation flow."""
    def __init__(self, username=None):
        self.transcript_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.current_transcript = ""
        self.transcript_lock = threading.Lock()
        self.full_conversation = []  # Store the entire conversation in memory

        # ===== MEMORY QUERY SUPPRESSION MECHANISM - TRANSCRIPT HANDLING =====
        # This is part of the bespoke solution to fix the "I don't know/Actually I do know" problem
        # When a memory-related query is detected, we need to suppress the initial incorrect response
        # in the transcript as well as in the audio playback.
        #
        # The approach works as follows:
        # 1. Set memory_retrieval_active flag when a memory query is detected
        # 2. Store responses in pending_responses instead of adding them directly to the transcript
        # 3. When memory_retrieval_active is set to False, we only add the last response (the correct one)
        #    to the transcript and discard the first response (the incorrect one)
        # It's critical for providing a good user experience with memory-related queries.
        self.memory_retrieval_active = False
        self.pending_responses = []  # Store responses during memory retrieval

        # Create base transcripts directory
        self.transcripts_dir = Path("gpt_realtime/gpt_transcripts")
        self.transcripts_dir.mkdir(exist_ok=True)

        # Initialize user-specific directory
        self.user_transcripts_dir = None
        logger.info(f"Base transcript directory set to: {self.transcripts_dir.absolute()}")

        # Set up user-specific directory immediately if username is provided
        self.set_user(username)

        # Track if speech has been detected
        self.speech_detected = False

        # Flag to track if transcript has been saved for this session
        self.transcript_saved = False
        # Store the saved transcript filename
        self.saved_transcript_file = None

    def set_user(self, username=None):
        """Set up user-specific transcript directory."""
        if username:
            # Create user-specific directory using sanitized username
            safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
            self.user_transcripts_dir = self.transcripts_dir / safe_username
            self.user_transcripts_dir.mkdir(exist_ok=True)
            logger.info(f"User transcript directory set to: {self.user_transcripts_dir.absolute()}")
        else:
            # Use 'anonymous' folder for users without username
            self.user_transcripts_dir = self.transcripts_dir / "anonymous"
            self.user_transcripts_dir.mkdir(exist_ok=True)
            logger.info(f"Anonymous transcript directory set to: {self.user_transcripts_dir.absolute()}")

    def start(self):
        """Start the transcript processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_transcripts, daemon=True)
            self.processing_thread.start()
            logger.info("Transcript processing thread started")

    def stop(self):
        """Stop the transcript processing thread."""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            # Add None to the queue to signal the thread to exit
            self.transcript_queue.put(None)
            self.processing_thread.join(timeout=2.0)
            logger.info("Transcript processing thread stopped")

    def add_transcript_event(self, event):
        """Add a transcript event to the processing queue."""
        self.transcript_queue.put(event)

    def add_speech_event(self, event_type):
        """Log speech started/stopped events to the transcript."""
        # Handle speech events
        if event_type == "speech_started" and not self.speech_detected:
            self.speech_detected = True
            with self.transcript_lock:
                self.full_conversation.append("--- Speech detected ---")
            logger.info("Speech started event added to transcript")

        elif event_type == "speech_stopped" and self.speech_detected:
            self.speech_detected = False
            with self.transcript_lock:
                # Add a placeholder for user speech that will be replaced
                # when the transcript comes in
                last_entries = self.full_conversation[-3:] if len(self.full_conversation) >= 3 else []
                if not any(entry.startswith("User:") for entry in last_entries):
                    # Only add this if we don't already have a User entry recently
                    self.full_conversation.append("User: [Processing speech...]")
                self.full_conversation.append("--- Speech ended ---")
                self.full_conversation.append("")  # Add blank line
            logger.info("Speech stopped event added to transcript")

    def get_current_transcript(self):
        """Get the current transcript with thread safety."""
        with self.transcript_lock:
            return self.current_transcript

    def clear_transcript(self):
        """Clear the current transcript when starting a new recording session."""
        with self.transcript_lock:
            self.current_transcript = ""
            # Reset speech detection
            self.speech_detected = False
            # Reset transcript saved flag
            self.transcript_saved = False
            self.saved_transcript_file = None

        logger.info("Transcript cleared for new recording session")

    def add_user_query(self, query_text):
        """Add a user's text query to the transcript."""
        if not query_text:
            return

        with self.transcript_lock:
            # Add to the in-memory transcript
            if self.current_transcript:
                self.current_transcript += f"\n\nUser: {query_text}"
            else:
                self.current_transcript = f"User: {query_text}"

            # Check if we need to replace a placeholder or add a new entry
            placeholder_replaced = False
            if self.full_conversation:
                # Look for speech processing placeholder to replace
                for i, entry in enumerate(self.full_conversation):
                    if entry == "User: [Processing speech...]":
                        self.full_conversation[i] = f"User: {query_text}"
                        placeholder_replaced = True
                        break

            # If no placeholder was found, add as a new entry
            if not placeholder_replaced:
                self.full_conversation.append(f"User: {query_text}")
                self.full_conversation.append("")  # Add blank line

        logger.info(f"Added user query to transcript: {query_text[:30]}...")

    def set_memory_retrieval_mode(self, active):
        """Set whether we're in memory retrieval mode.

        This is part of the memory query suppression mechanism that prevents the system
        from showing the initial incorrect response in the transcript for memory-related queries.

        When exiting memory retrieval mode, we process any pending responses but skip the first one
        (which is the incorrect response) and only keep the last one (the corrected response).

        Args:
            active (bool): Whether to activate memory retrieval mode
        """
        with self.transcript_lock:
            self.memory_retrieval_active = active
            if not active and self.pending_responses:
                # ===== MEMORY QUERY SUPPRESSION - TRANSCRIPT HANDLING =====
                # This is where we actually suppress the incorrect response in the transcript.
                # When exiting memory retrieval mode, we only add the last response (the corrected one)
                # to the transcript and discard the first response (the incorrect one).
                #
                # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
                # It's critical for providing a good user experience with memory-related queries.
                if len(self.pending_responses) > 1:
                    # Skip the first response and only keep the corrected one
                    correct_response = self.pending_responses[-1]
                    self.full_conversation.append(f"Assistant: {correct_response}")
                    self.full_conversation.append("")  # Add blank line
                    logger.info(f"Added corrected memory response to transcript: {correct_response[:30]}...")
                self.pending_responses = []

    def add_assistant_response(self, response_text):
        """Add an assistant's text response to the transcript."""
        if not response_text:
            return

        with self.transcript_lock:
            # Add to the in-memory transcript
            if self.current_transcript:
                self.current_transcript += f"\n\nAssistant: {response_text}"
            else:
                self.current_transcript = f"Assistant: {response_text}"

            # ===== MEMORY QUERY SUPPRESSION - RESPONSE HANDLING =====
            # This is part of the memory query suppression mechanism.
            # When in memory retrieval mode, we don't add responses directly to the transcript.
            # Instead, we store them in pending_responses and process them when exiting memory retrieval mode.
            #
            # This allows us to skip the first response (which is incorrect) and only keep the last one
            # (which has the correct memory context).
            #
            # DO NOT REMOVE OR MODIFY THIS MECHANISM without understanding the implications!
            # It's critical for providing a good user experience with memory-related queries.
            if self.memory_retrieval_active:
                self.pending_responses.append(response_text)
                logger.info(f"Stored assistant response during memory retrieval: {response_text[:30]}...")
                return

            # Add as a new entry
            self.full_conversation.append(f"Assistant: {response_text}")
            self.full_conversation.append("")  # Add blank line

        logger.info(f"Added assistant response to transcript: {response_text[:30]}...")

    def update_transcript(self):
        """Update the in-memory transcript without saving to a file.
        Returns a cleaned version of the conversation for display purposes."""
        if not self.full_conversation:
            logger.info("No transcript to update")
            return None

        # Clean up any unresolved placeholders
        cleaned_conversation = []
        for entry in self.full_conversation:
            if entry == "User: [Processing speech...]":
                # Skip unresolved placeholders
                continue
            cleaned_conversation.append(entry)

        logger.info("In-memory transcript updated")
        return cleaned_conversation

    def get_last_assistant_response(self):
        """Get the last assistant response from the transcript.

        Returns:
            str: The last assistant response, or an empty string if none found.
        """
        if not self.full_conversation:
            logger.info("No transcript to get assistant response from")
            return ""

        # Look for the last assistant response in the full conversation
        for entry in reversed(self.full_conversation):
            if entry.startswith("Assistant: "):
                response = entry[len("Assistant: "):]
                logger.info(f"Found last assistant response: {response[:30]}...")
                return response

        # If we're in memory retrieval mode, check pending responses
        if self.memory_retrieval_active and self.pending_responses:
            response = self.pending_responses[-1]
            logger.info(f"Found last assistant response in pending responses: {response[:30]}...")
            return response

        logger.info("No assistant response found in transcript")
        return ""

    def save_transcript(self, force=False):
        """Save the complete transcript to a file at the end of the conversation.

        Args:
            force (bool): If True, always save the transcript. If False, only save if this is the final save.
        """
        if not self.full_conversation:
            logger.info("No transcript to save")
            return None

        if not force:
            logger.info("Skipping intermediate transcript save")
            return None

        # If transcript has already been saved for this session, return the existing filename
        if self.transcript_saved and self.saved_transcript_file:
            logger.info(f"Transcript already saved to: {self.saved_transcript_file}")
            return self.saved_transcript_file

        # Clean up any unresolved placeholders
        cleaned_conversation = []
        for entry in self.full_conversation:
            if entry == "User: [Processing speech...]":
                continue
            cleaned_conversation.append(entry)

        # Ensure we have a directory to save to (use base dir if user dir not set)
        save_dir = self.user_transcripts_dir or self.transcripts_dir

        # Create a new file for this conversation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"conversation_{timestamp}.txt"

        try:
            # Use UTF-8 encoding to handle non-ASCII characters
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Complete Conversation Transcript: {timestamp}\n")
                f.write("-" * 50 + "\n\n")

                # Write all conversation lines
                for line in cleaned_conversation:
                    f.write(f"{line}\n")

            logger.info(f"Complete transcript saved to: {filename}")

            # Mark transcript as saved for this session
            self.transcript_saved = True
            self.saved_transcript_file = filename

            return filename
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            return None

    def _process_transcripts(self):
        """Process transcript events from the queue."""
        while self.running:
            try:
                # Block for a short time to avoid busy waiting
                event = self.transcript_queue.get(timeout=0.5)

                # Check for exit signal
                if event is None:
                    break

                event_type = event.get("type")

                # logger.debug(f"Processing event type: {event_type}")

                # Process different transcript event types
                if event_type == "response.audio_transcript.delta":
                    delta = event.get("delta", "")
                    if delta:
                        with self.transcript_lock:
                            # Only prepend "Assistant: " if this is the first part of the response
                            if not self.current_transcript or self.current_transcript.endswith("\n\nUser: "):
                                # For in-memory tracking
                                if not self.current_transcript:
                                    self.current_transcript = f"Assistant: {delta}"
                                else:
                                    self.current_transcript += f"\n\nAssistant: {delta}"

                                # For full conversation
                                self.full_conversation.append(f"Assistant: {delta}")
                            else:
                                # Continuation of assistant's response
                                self.current_transcript += delta

                                # Update the last line in full conversation
                                if self.full_conversation and self.full_conversation[-1].startswith("Assistant:"):
                                    self.full_conversation[-1] += delta

                        # logger.debug(f"Transcript delta processed")

                elif event_type == "response.audio_transcript.done":
                    transcript = event.get("transcript", "")
                    if transcript:
                        with self.transcript_lock:
                            # Append the assistant's response to the existing transcript
                            if "User: " in self.current_transcript:
                                self.current_transcript += f"\n\nAssistant: {transcript}"
                            else:
                                self.current_transcript = f"Assistant: {transcript}"

                            # Append to full conversation
                            self.full_conversation.append(f"Assistant: {transcript}")
                            self.full_conversation.append("------------------------------")
                            self.full_conversation.append("")  # Add blank line

                        logger.info("Final transcript processed")
                        # logger.debug(f"Processing response.audio_transcript.done event with transcript: {transcript}")
                        # logger.debug(f"Updated full conversation with assistant response: {transcript}")

                # Mark task as done
                self.transcript_queue.task_done()

                # logger.debug(f"Event processed: {event_type}")

                # logger.debug(f"Current transcript before processing: {self.current_transcript}")
                # logger.debug(f"Current transcript after processing: {self.current_transcript}")

            except queue.Empty:
                # Queue timeout, continue looping
                continue
            except Exception as e:
                logger.error(f"Error processing transcript: {e}")
                import traceback
                logger.error(traceback.format_exc())
