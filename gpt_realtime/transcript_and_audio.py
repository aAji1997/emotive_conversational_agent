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


class TranscriptProcessor:
    """Processes transcripts in a separate thread to avoid blocking the main conversation flow."""
    def __init__(self):
        self.transcript_queue = queue.Queue()
        self.processing_thread = None
        self.running = False
        self.current_transcript = ""
        self.transcript_lock = threading.Lock()
        self.full_conversation = []  # Store the entire conversation in memory

        # Create transcripts directory if it doesn't exist
        self.transcripts_dir = Path("gpt_realtime/gpt_transcripts")
        self.transcripts_dir.mkdir(exist_ok=True)
        logger.info(f"Transcript directory set to: {self.transcripts_dir.absolute()}")

        # Track if speech has been detected
        self.speech_detected = False

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

    def save_transcript(self, force=False):
        """Save the complete transcript to a file at the end of the conversation.

        Args:
            force (bool): If True, always save the transcript. If False, only save if this is the final save.
        """
        if not self.full_conversation:
            logger.info("No transcript to save")
            return None

        # Only save if force=True (this is the final save at the end of the session)
        # This prevents saving intermediate transcripts during the conversation
        if not force:
            logger.info("Skipping intermediate transcript save")
            return None

        # Clean up any unresolved placeholders before saving
        cleaned_conversation = []
        for entry in self.full_conversation:
            if entry == "User: [Processing speech...]":
                # Skip unresolved placeholders
                continue
            cleaned_conversation.append(entry)

        # Create a new file for this conversation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.transcripts_dir / f"conversation_{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write(f"Complete Conversation Transcript: {timestamp}\n")
                f.write("-" * 50 + "\n\n")

                # Write all conversation lines
                for line in cleaned_conversation:
                    f.write(f"{line}\n")

            logger.info(f"Complete transcript saved to: {filename}")
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
