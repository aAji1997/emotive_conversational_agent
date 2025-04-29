"""
Shared Emotion State Manager for in-memory emotion data sharing between processes.

This module provides a thread-safe wrapper around multiprocessing shared dictionaries
for storing and retrieving emotion scores. It ensures proper synchronization between
the sentiment analysis process and the visualization process.
"""

import threading
import logging
import time
from typing import Dict, Optional, Tuple, Any

# Completely disable all logging for this module
logging.getLogger('gpt_realtime.shared_emotion_state').setLevel(logging.CRITICAL)

# Configure logging
logger = logging.getLogger(__name__)

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

class SharedEmotionState:
    """
    A thread-safe wrapper around multiprocessing shared dictionaries for emotion scores.

    This class provides methods for updating and retrieving emotion scores in a thread-safe
    manner, ensuring proper synchronization between processes.
    """

    def __init__(self, shared_dict=None):
        """
        Initialize the shared emotion state.

        Args:
            shared_dict: A multiprocessing.Manager().dict() object for sharing data between processes.
                         If None, this will be a local-only state (not shared between processes).
        """
        self.lock = threading.RLock()  # Reentrant lock for thread safety

        # Use the provided shared dictionary or create a local one
        self.shared_dict = shared_dict if shared_dict is not None else {
            'user': {emotion: 0.0 for emotion in CORE_EMOTIONS},
            'assistant': {emotion: 0.0 for emotion in CORE_EMOTIONS},
            'last_update_time': time.time(),
            'assistant_is_responding': False,  # Flag to track if assistant is responding
            'silence_detected': False,  # Flag to track if silence is detected
            'last_audio_activity_time': time.time(),  # Track the last time audio activity was detected
            'vad_speech_detected': False,  # Flag to track if VAD has detected speech
            'is_audio_mode': False  # Flag to track if we're in audio mode or text chat mode
        }

        # Initialize the shared dictionary if it doesn't have the expected structure
        with self.lock:
            if 'user' not in self.shared_dict:
                self.shared_dict['user'] = {emotion: 0.0 for emotion in CORE_EMOTIONS}
            if 'assistant' not in self.shared_dict:
                self.shared_dict['assistant'] = {emotion: 0.0 for emotion in CORE_EMOTIONS}
            if 'last_update_time' not in self.shared_dict:
                self.shared_dict['last_update_time'] = time.time()
            if 'assistant_is_responding' not in self.shared_dict:
                self.shared_dict['assistant_is_responding'] = False
            if 'silence_detected' not in self.shared_dict:
                self.shared_dict['silence_detected'] = False
            if 'last_audio_activity_time' not in self.shared_dict:
                self.shared_dict['last_audio_activity_time'] = time.time()
            if 'vad_speech_detected' not in self.shared_dict:
                self.shared_dict['vad_speech_detected'] = False
            if 'is_audio_mode' not in self.shared_dict:
                self.shared_dict['is_audio_mode'] = False

    def update_emotion_scores(self, source: str, emotion_scores: Dict[str, float]) -> bool:
        """
        Update the emotion scores for the specified source.

        Args:
            source: Either 'user' or 'assistant'.
            emotion_scores: A dictionary mapping emotion names to scores.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if source not in ['user', 'assistant']:
            return False

        try:
            with self.lock:
                # Ultra-efficient version - just update the scores directly
                new_scores = dict(self.shared_dict.get(source, {}))

                # Update with new scores
                for emotion, score in emotion_scores.items():
                    if emotion in CORE_EMOTIONS:
                        new_scores[emotion] = float(score)

                # Replace the entire dictionary
                self.shared_dict[source] = new_scores

                # Update the last update time
                self.shared_dict['last_update_time'] = time.time()

                return True
        except:
            return False

    def get_emotion_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get the current emotion scores for both user and assistant.

        Returns:
            tuple: (user_scores, assistant_scores) dictionaries.
        """
        with self.lock:
            # Ultra-efficient version - just return copies of the dictionaries
            user_scores = dict(self.shared_dict.get('user', {}))
            assistant_scores = dict(self.shared_dict.get('assistant', {}))
            return user_scores, assistant_scores

    def get_last_update_time(self) -> float:
        """
        Get the timestamp of the last update.

        Returns:
            float: The timestamp of the last update.
        """
        with self.lock:
            return self.shared_dict.get('last_update_time', time.time())

    def is_shared(self) -> bool:
        """
        Check if this state is using a shared dictionary.

        Returns:
            bool: True if using a shared dictionary, False otherwise.
        """
        # Check if the shared_dict is a Manager.dict() by checking for the _callmethod attribute
        return hasattr(self.shared_dict, '_callmethod')

    def set_assistant_responding(self, is_responding: bool) -> bool:
        """
        Set the flag indicating whether the assistant is currently responding.

        Args:
            is_responding: True if the assistant is responding, False otherwise.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with self.lock:
                self.shared_dict['assistant_is_responding'] = bool(is_responding)
                return True
        except:
            return False

    def is_assistant_responding(self) -> bool:
        """
        Check if the assistant is currently responding.

        Returns:
            bool: True if the assistant is responding, False otherwise.
        """
        with self.lock:
            return bool(self.shared_dict.get('assistant_is_responding', False))

    def set_silence_detected(self, is_silence: bool) -> bool:
        """
        Set the flag indicating whether silence is detected.

        Args:
            is_silence: True if silence is detected, False otherwise.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with self.lock:
                self.shared_dict['silence_detected'] = bool(is_silence)
                return True
        except:
            return False

    def is_silence_detected(self) -> bool:
        """
        Check if silence is currently detected.

        Returns:
            bool: True if silence is detected, False otherwise.
        """
        with self.lock:
            return bool(self.shared_dict.get('silence_detected', False))

    def update_audio_activity_time(self) -> bool:
        """
        Update the timestamp of the last audio activity.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with self.lock:
                self.shared_dict['last_audio_activity_time'] = time.time()
                return True
        except:
            return False

    def get_time_since_last_audio_activity(self) -> float:
        """
        Get the time elapsed since the last audio activity in seconds.

        Returns:
            float: Time in seconds since the last audio activity.
        """
        with self.lock:
            last_time = self.shared_dict.get('last_audio_activity_time', time.time())
            return time.time() - last_time

    def set_vad_speech_detected(self, speech_detected: bool) -> bool:
        """
        Set the flag indicating whether VAD has detected speech.

        Args:
            speech_detected: True if speech is detected, False otherwise.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with self.lock:
                self.shared_dict['vad_speech_detected'] = bool(speech_detected)
                return True
        except:
            return False

    def is_vad_speech_detected(self) -> bool:
        """
        Check if VAD has detected speech.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        with self.lock:
            return bool(self.shared_dict.get('vad_speech_detected', False))

    def set_audio_mode(self, is_audio_mode: bool) -> bool:
        """
        Set the flag indicating whether we're in audio mode.

        Args:
            is_audio_mode: True if in audio mode, False if in text chat mode.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with self.lock:
                self.shared_dict['is_audio_mode'] = bool(is_audio_mode)
                return True
        except:
            return False

    def is_audio_mode(self) -> bool:
        """
        Check if we're in audio mode.

        Returns:
            bool: True if in audio mode, False if in text chat mode.
        """
        with self.lock:
            return bool(self.shared_dict.get('is_audio_mode', False))
