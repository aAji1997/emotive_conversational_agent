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
            'last_update_time': time.time()
        }
        
        # Initialize the shared dictionary if it doesn't have the expected structure
        with self.lock:
            if 'user' not in self.shared_dict:
                self.shared_dict['user'] = {emotion: 0.0 for emotion in CORE_EMOTIONS}
            if 'assistant' not in self.shared_dict:
                self.shared_dict['assistant'] = {emotion: 0.0 for emotion in CORE_EMOTIONS}
            if 'last_update_time' not in self.shared_dict:
                self.shared_dict['last_update_time'] = time.time()
    
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
            logger.error(f"Invalid source: {source}. Must be 'user' or 'assistant'.")
            return False
        
        try:
            with self.lock:
                # For multiprocessing shared dictionaries, we need to replace the entire dictionary
                # Create a new dictionary with all emotion scores
                new_scores = {emotion: 0.0 for emotion in CORE_EMOTIONS}
                
                # Update with existing scores first (if available)
                if source in self.shared_dict:
                    existing_scores = self.shared_dict[source]
                    if isinstance(existing_scores, dict):
                        new_scores.update(dict(existing_scores))
                
                # Then update with new scores
                for emotion, score in emotion_scores.items():
                    if emotion in CORE_EMOTIONS:
                        new_scores[emotion] = float(score)  # Ensure scores are float
                
                # Replace the entire dictionary in the shared dictionary
                self.shared_dict[source] = new_scores
                
                # Update the last update time
                self.shared_dict['last_update_time'] = time.time()
                
                logger.debug(f"Updated {source} emotion scores: {new_scores}")
                return True
        except Exception as e:
            logger.error(f"Error updating emotion scores: {e}")
            return False
    
    def get_emotion_scores(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get the current emotion scores for both user and assistant.
        
        Returns:
            tuple: (user_scores, assistant_scores) dictionaries.
        """
        with self.lock:
            # Create copies of the dictionaries to avoid modification issues
            user_scores = dict(self.shared_dict.get('user', {emotion: 0.0 for emotion in CORE_EMOTIONS}))
            assistant_scores = dict(self.shared_dict.get('assistant', {emotion: 0.0 for emotion in CORE_EMOTIONS}))
            
            # Ensure all emotions are present
            for emotion in CORE_EMOTIONS:
                if emotion not in user_scores:
                    user_scores[emotion] = 0.0
                if emotion not in assistant_scores:
                    assistant_scores[emotion] = 0.0
            
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
