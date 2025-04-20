"""
Emotion Transition Manager for smooth transitions between emotional states.

This module provides functionality for smoothly transitioning between emotional states
using interpolation and easing functions. It maintains a history of emotional states
and provides methods for retrieving interpolated states for visualization.
"""

import time
import numpy as np
import json
import os
from collections import deque
from scipy.interpolate import CubicSpline

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

class EmotionTransitionManager:
    """
    A class for managing smooth transitions between emotional states.
    """

    def __init__(self, transition_speed=0.3, history_length=10, user_id=None):
        """
        Initialize the emotion transition manager.

        Args:
            transition_speed (float): Speed of transitions between emotional states (0-1).
            history_length (int): Number of historical emotional states to maintain.
            user_id (str, optional): User ID for saving emotion data to user-specific files.
        """
        self.transition_speed = transition_speed
        self.history_length = history_length
        self.user_id = user_id

        # Initialize current emotion states
        self.current_user_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}
        self.current_assistant_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}

        # Initialize target emotion states (for smooth transitions)
        self.target_user_emotions = self.current_user_emotions.copy()
        self.target_assistant_emotions = self.current_assistant_emotions.copy()

        # Initialize emotion history
        self.user_emotion_history = deque(maxlen=history_length)
        self.assistant_emotion_history = deque(maxlen=history_length)

        # Add initial states to history
        self.user_emotion_history.append((time.time(), self.current_user_emotions.copy()))
        self.assistant_emotion_history.append((time.time(), self.current_assistant_emotions.copy()))

        # Last update time for controlling animation speed
        self.last_update_time = time.time()

        # Create directory for saving emotion data
        self.temp_dir = self._get_temp_dir()

    def _get_temp_dir(self):
        """Get the temporary directory for saving emotion data."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if self.user_id:
            # Use user-specific directory if user_id is provided
            temp_dir = os.path.join(base_dir, 'user_data', self.user_id)
        else:
            # Use default temp directory
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')

        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def update_emotions(self, user_emotions=None, assistant_emotions=None):
        """
        Update the target emotion states.

        Args:
            user_emotions (dict, optional): New user emotion scores.
            assistant_emotions (dict, optional): New assistant emotion scores.
        """
        current_time = time.time()

        if user_emotions:
            # Ensure all emotions are present
            for emotion in CORE_EMOTIONS:
                if emotion not in user_emotions:
                    user_emotions[emotion] = 0.0

            # Convert integer scores to float for smoother transitions
            user_emotions = {k: float(v) for k, v in user_emotions.items()}

            # Update target emotions
            self.target_user_emotions = user_emotions.copy()

            # Add to history
            self.user_emotion_history.append((current_time, user_emotions.copy()))

        if assistant_emotions:
            # Ensure all emotions are present
            for emotion in CORE_EMOTIONS:
                if emotion not in assistant_emotions:
                    assistant_emotions[emotion] = 0.0

            # Convert integer scores to float for smoother transitions
            assistant_emotions = {k: float(v) for k, v in assistant_emotions.items()}

            # Update target emotions
            self.target_assistant_emotions = assistant_emotions.copy()

            # Add to history
            self.assistant_emotion_history.append((current_time, assistant_emotions.copy()))

        # Save to file for visualization
        self._save_current_emotions()

    def _save_current_emotions(self):
        """Save current emotion states to file for visualization."""
        # Create a temporary file path
        temp_file = os.path.join(self.temp_dir, 'current_emotion_scores.json')

        # Prepare data
        current_scores = {
            'user': self.target_user_emotions,
            'assistant': self.target_assistant_emotions,
            'timestamp': time.time()
        }

        # Save to file
        with open(temp_file, 'w') as f:
            json.dump(current_scores, f)

        # Also save to a conversation-specific file if user_id is provided
        if self.user_id:
            # Create a conversation-specific file
            conversation_file = os.path.join(self.temp_dir, 'conversation_emotions.json')

            # Load existing data if available
            conversation_data = []
            if os.path.exists(conversation_file):
                try:
                    with open(conversation_file, 'r') as f:
                        conversation_data = json.load(f)
                except Exception:
                    conversation_data = []

            # Add new data
            conversation_data.append({
                'user': self.target_user_emotions,
                'assistant': self.target_assistant_emotions,
                'timestamp': time.time()
            })

            # Save to file
            with open(conversation_file, 'w') as f:
                json.dump(conversation_data, f)

    def get_current_emotions(self):
        """
        Get the current interpolated emotion states.

        Returns:
            tuple: (user_emotions, assistant_emotions) dictionaries.
        """
        # Interpolate emotions for smooth transitions
        self._interpolate_emotions()

        return self.current_user_emotions.copy(), self.current_assistant_emotions.copy()

    def _interpolate_emotions(self):
        """Interpolate between current and target emotion states for smooth transitions."""
        # Calculate time-based interpolation factor
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Calculate interpolation factor with ease-in-out function
        # This creates a more natural, smooth transition
        alpha = min(1.0, dt / self.transition_speed)
        alpha = self._ease_in_out(alpha)

        # Interpolate user emotions
        for emotion in CORE_EMOTIONS:
            self.current_user_emotions[emotion] += alpha * (
                self.target_user_emotions[emotion] - self.current_user_emotions[emotion]
            )

            self.current_assistant_emotions[emotion] += alpha * (
                self.target_assistant_emotions[emotion] - self.current_assistant_emotions[emotion]
            )

        # Apply a small amount of smoothing between adjacent emotions for continuity
        # This creates a more continuous emotional space by slightly blending neighboring emotions
        smoothed_user = self.current_user_emotions.copy()
        smoothed_assistant = self.current_assistant_emotions.copy()

        # Define smoothing factor (small value to maintain the primary emotion scores)
        smooth_factor = 0.15

        # Apply smoothing for each emotion based on its neighbors
        for i, emotion in enumerate(CORE_EMOTIONS):
            # Get indices of adjacent emotions (circular list)
            prev_idx = (i - 1) % len(CORE_EMOTIONS)
            next_idx = (i + 1) % len(CORE_EMOTIONS)
            prev_emotion = CORE_EMOTIONS[prev_idx]
            next_emotion = CORE_EMOTIONS[next_idx]

            # Apply smoothing to user emotions
            smoothed_user[emotion] = (1 - smooth_factor) * self.current_user_emotions[emotion] + \
                                    (smooth_factor / 2) * self.current_user_emotions[prev_emotion] + \
                                    (smooth_factor / 2) * self.current_user_emotions[next_emotion]

            # Apply smoothing to assistant emotions
            smoothed_assistant[emotion] = (1 - smooth_factor) * self.current_assistant_emotions[emotion] + \
                                         (smooth_factor / 2) * self.current_assistant_emotions[prev_emotion] + \
                                         (smooth_factor / 2) * self.current_assistant_emotions[next_emotion]

        # Update the current emotions with the smoothed values
        self.current_user_emotions = smoothed_user
        self.current_assistant_emotions = smoothed_assistant

    def _ease_in_out(self, t):
        """
        Cubic ease-in-out function for smooth transitions.

        Args:
            t (float): Input value between 0 and 1.

        Returns:
            float: Eased value between 0 and 1.
        """
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2

    def _get_interpolated_polygon_points(self, emotion_scores, num_points=24):
        """
        Generate a smooth polygon with more points for continuous visualization.

        Args:
            emotion_scores (dict): Dictionary mapping emotions to scores.
            num_points (int): Number of points to generate for the polygon.

        Returns:
            tuple: (r_values, theta_values) for the polygon points.
        """
        # Get the base points for the 8 core emotions
        base_r = []
        base_theta = []

        for i, emotion in enumerate(CORE_EMOTIONS):
            angle_deg = i * 45
            score = emotion_scores[emotion]
            base_r.append(score)
            base_theta.append(angle_deg)

        # Add the first point again to close the loop
        base_r.append(base_r[0])
        base_theta.append(base_theta[0])

        # Convert to numpy arrays for interpolation
        base_r = np.array(base_r)
        base_theta = np.array(base_theta)

        # Generate more points using cubic spline interpolation
        # This creates a smooth curve through the core emotion points
        t = np.linspace(0, 1, len(base_r))
        t_interp = np.linspace(0, 1, num_points)

        # Convert to Cartesian coordinates for interpolation
        x = base_r * np.cos(np.radians(base_theta))
        y = base_r * np.sin(np.radians(base_theta))

        # Interpolate x and y separately
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')

        # Generate interpolated points
        x_interp = cs_x(t_interp)
        y_interp = cs_y(t_interp)

        # Convert back to polar coordinates
        r_interp = np.sqrt(x_interp**2 + y_interp**2)
        theta_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360

        return r_interp, theta_interp

    def get_emotion_history(self):
        """
        Get the emotion history.

        Returns:
            tuple: (user_history, assistant_history) lists of (timestamp, emotions) tuples.
        """
        return list(self.user_emotion_history), list(self.assistant_emotion_history)

    def save_emotion_data(self, file_path=None):
        """
        Save the emotion data to a file.

        Args:
            file_path (str, optional): Path to save the emotion data.
                If not provided, a default path will be used.

        Returns:
            str: Path to the saved file.
        """
        if file_path is None:
            # Generate a default file path
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"emotion_data_{timestamp}.json"

            if self.user_id:
                file_path = os.path.join(self.temp_dir, file_name)
            else:
                file_path = os.path.join(self.temp_dir, file_name)

        # Prepare data
        data = {
            'user_history': [
                {'timestamp': ts, 'emotions': emotions}
                for ts, emotions in self.user_emotion_history
            ],
            'assistant_history': [
                {'timestamp': ts, 'emotions': emotions}
                for ts, emotions in self.assistant_emotion_history
            ],
            'metadata': {
                'timestamp': time.time(),
                'user_id': self.user_id
            }
        }

        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Emotion data saved to {file_path}")
        return file_path
