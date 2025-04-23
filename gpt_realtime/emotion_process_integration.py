"""
Emotion Process Integration

This module provides a simple integration between the standalone emotion visualization
and the main application.
"""

import subprocess
import time
import threading
import logging
import os
import signal
import sys
import json
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

class EmotionProcessIntegration:
    """
    Class for integrating the standalone emotion visualization with the main application.
    """

    def __init__(self, shared_emotion_scores=None, data_file=None, update_interval=0.1):
        """
        Initialize the emotion process integration.

        Args:
            shared_emotion_scores: Optional shared dictionary for emotion scores
            data_file: Path to the emotion data file
            update_interval: Interval between updates in seconds
        """
        # Store the shared emotion scores
        self.shared_emotion_scores = shared_emotion_scores

        # Set up the data file
        self.data_file = data_file or Path("emotion_data.json")

        # Set up the update interval
        self.update_interval = update_interval

        # Initialize current emotion states
        self.current_user_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}
        self.current_assistant_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}

        # Process for the visualization
        self.process = None

        # Create a thread for updating the file
        self.update_thread = None
        self.stop_event = threading.Event()

        # Flag to track if the integration is running
        self.is_running = False

        # Last update time for rate limiting
        self.last_update_time = time.time()

        # Create the data file with initial data
        self._write_data()

    def _write_data(self):
        """Write the current emotion data to the file."""
        try:
            # Create the data dictionary
            data = {
                'user': self.current_user_emotions,
                'assistant': self.current_assistant_emotions,
                'timestamp': time.time()
            }

            # Write the data to the file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Wrote emotion data to {self.data_file}")
            return True
        except Exception as e:
            logger.error(f"Error writing emotion data: {e}")
            return False

    def _update_loop(self):
        """Update loop for writing emotion data to the file."""
        logger.info(f"Starting emotion process integration update loop, writing to {self.data_file}")

        while not self.stop_event.is_set():
            try:
                # Rate limit updates to avoid excessive file I/O
                current_time = time.time()
                if current_time - self.last_update_time < self.update_interval:
                    # Sleep for a short time to avoid busy waiting
                    time.sleep(0.01)
                    continue

                self.last_update_time = current_time

                # Update from shared emotion scores if available
                if self.shared_emotion_scores is not None:
                    try:
                        # If it's a SharedEmotionState instance
                        if hasattr(self.shared_emotion_scores, 'get_emotion_scores'):
                            user_emotions, assistant_emotions = self.shared_emotion_scores.get_emotion_scores()

                            # Only update if we have valid emotion scores
                            if user_emotions and isinstance(user_emotions, dict):
                                self.current_user_emotions = user_emotions.copy()

                            if assistant_emotions and isinstance(assistant_emotions, dict):
                                self.current_assistant_emotions = assistant_emotions.copy()

                        # If it's a dictionary
                        elif isinstance(self.shared_emotion_scores, dict):
                            user_emotions = self.shared_emotion_scores.get('user', {})
                            assistant_emotions = self.shared_emotion_scores.get('assistant', {})

                            # Only update if we have valid emotion scores
                            if user_emotions and isinstance(user_emotions, dict):
                                self.current_user_emotions = user_emotions.copy()

                            if assistant_emotions and isinstance(assistant_emotions, dict):
                                self.current_assistant_emotions = assistant_emotions.copy()
                    except Exception as e:
                        logger.error(f"Error getting emotion scores: {e}")

                # Write the data to the file
                self._write_data()

            except Exception as e:
                logger.error(f"Error in emotion process integration update loop: {e}")
                time.sleep(0.5)  # Sleep longer on error

        logger.info("Emotion process integration update loop stopped")

    def start(self):
        """Start the emotion process integration."""
        if self.is_running:
            logger.warning("Emotion process integration is already running")
            return False

        # Create a multiprocessing manager if needed
        if self.shared_emotion_scores is None:
            try:
                self.manager = multiprocessing.Manager()
                self.shared_dict = self.manager.dict({
                    'user': {emotion: 0.0 for emotion in CORE_EMOTIONS},
                    'assistant': {emotion: 0.0 for emotion in CORE_EMOTIONS},
                    'last_update_time': time.time()
                })

                # Import the SharedEmotionState class
                from gpt_realtime.shared_emotion_state import SharedEmotionState

                # Create a shared emotion state
                self.shared_emotion_scores = SharedEmotionState(self.shared_dict)
                logger.info("Created new SharedEmotionState with multiprocessing.Manager().dict()")
            except Exception as e:
                logger.error(f"Error creating shared emotion state: {e}")
                return False

        # Start the visualization process
        try:
            # Get the path to the standalone visualization script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            viz_script = os.path.join(os.path.dirname(script_dir), "emotion_viz_standalone_memory.py")

            # Check if the script exists
            if not os.path.exists(viz_script):
                logger.error(f"Visualization script not found: {viz_script}")
                return False

            # Start the visualization in a separate process
            self.process = subprocess.Popen(
                [sys.executable, viz_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, 'PYTHONPATH': f"{os.environ.get('PYTHONPATH', '')}:{os.path.dirname(script_dir)}"}
            )

            # Check if the process started successfully
            if self.process.poll() is not None:
                logger.error(f"Failed to start visualization process: {self.process.stderr.read()}")
                return False

            logger.info("Emotion process integration started")
            print("\n[EMOTION WHEEL] Standalone emotion visualization started with in-memory sharing")
            print("[EMOTION WHEEL] Access the visualization at: http://127.0.0.1:8050/")

            self.is_running = True
            return True
        except Exception as e:
            logger.error(f"Error starting emotion process integration: {e}")
            return False

    def stop(self):
        """Stop the emotion process integration."""
        if not self.is_running:
            logger.warning("Emotion process integration is not running")
            return False

        # Stop the visualization process
        try:
            if self.process:
                # Try to terminate the process gracefully
                self.process.terminate()

                # Wait for the process to terminate
                self.process.wait(timeout=5)

                # If the process is still running, kill it
                if self.process.poll() is None:
                    self.process.kill()

                self.process = None

            # Shut down the manager if we created it
            if self.manager is not None:
                self.manager.shutdown()
                self.manager = None
                self.shared_dict = None

            logger.info("Emotion process integration stopped")
            self.is_running = False
            return True
        except Exception as e:
            logger.error(f"Error stopping emotion process integration: {e}")
            return False

    def update_emotions(self, user_emotions=None, assistant_emotions=None):
        """
        Update the emotion data.

        Args:
            user_emotions: Dictionary of user emotion scores
            assistant_emotions: Dictionary of assistant emotion scores
        """
        # Update local copies
        if user_emotions:
            self.current_user_emotions = user_emotions.copy()

        if assistant_emotions:
            self.current_assistant_emotions = assistant_emotions.copy()

        # Update shared emotion scores if available
        if self.shared_emotion_scores is not None:
            try:
                # If it's a SharedEmotionState instance
                if hasattr(self.shared_emotion_scores, 'update_emotion_scores'):
                    if user_emotions:
                        self.shared_emotion_scores.update_emotion_scores('user', user_emotions)
                    if assistant_emotions:
                        self.shared_emotion_scores.update_emotion_scores('assistant', assistant_emotions)
                # If it's a dictionary
                elif isinstance(self.shared_emotion_scores, dict):
                    if user_emotions:
                        self.shared_emotion_scores['user'] = user_emotions.copy()
                    if assistant_emotions:
                        self.shared_emotion_scores['assistant'] = assistant_emotions.copy()
            except Exception as e:
                logger.error(f"Error updating shared emotion scores: {e}")

        # Write the data to the file
        self._write_data()

        return True
