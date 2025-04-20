"""
Emotion Wheel Visualization based on Plutchik's wheel of emotions.

This module provides a continuous emotional space visualization that represents
emotional states as a continuous contour on Plutchik's wheel of emotions.
It supports smooth transitions between emotional states and overlaying
multiple emotional states (e.g., user and assistant).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import colorsys
import time
from scipy.interpolate import CubicSpline

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

# Define colors for each emotion (using Plutchik's traditional color scheme)
EMOTION_COLORS = {
    "Joy": "#FFFF00",  # Yellow
    "Trust": "#00FF00",  # Green
    "Fear": "#00FFFF",  # Light Blue
    "Surprise": "#0000FF",  # Blue
    "Sadness": "#FF00FF",  # Purple
    "Disgust": "#800080",  # Violet
    "Anger": "#FF0000",  # Red
    "Anticipation": "#FFA500",  # Orange
}

class EmotionWheelVisualization:
    """
    A class for creating and updating a continuous emotional space visualization
    based on Plutchik's wheel of emotions.
    """

    def __init__(self, resolution=100, transition_speed=0.3):
        """
        Initialize the emotion wheel visualization.

        Args:
            resolution (int): The resolution of the emotion wheel grid.
            transition_speed (float): Speed of transitions between emotional states (0-1).
        """
        self.resolution = resolution
        self.transition_speed = transition_speed

        # Initialize current emotion states
        self.current_user_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}
        self.current_assistant_emotions = {emotion: 0.0 for emotion in CORE_EMOTIONS}

        # Initialize target emotion states (for smooth transitions)
        self.target_user_emotions = self.current_user_emotions.copy()
        self.target_assistant_emotions = self.current_assistant_emotions.copy()

        # Initialize the figure
        self.fig = self._create_base_figure()

        # Last update time for controlling animation speed
        self.last_update_time = time.time()

    def _create_base_figure(self):
        """Create the base figure for the emotion wheel."""
        # Create a figure with a polar subplot
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])

        # Add the emotion wheel background
        self._add_emotion_wheel_background(fig)

        # Configure the layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 3],
                    showticklabels=False,
                    ticks='',
                ),
                angularaxis=dict(
                    visible=True,
                    tickmode='array',
                    tickvals=np.linspace(0, 315, 8),
                    ticktext=CORE_EMOTIONS,
                    direction='clockwise',
                    rotation=22.5,  # Offset to align with emotion sectors
                ),
                bgcolor='rgba(240, 240, 240, 0.8)',
            ),
            showlegend=True,
            title="Emotional GPS - Continuous Emotional Space",
            height=600,
            width=700,
            margin=dict(l=50, r=50, t=100, b=50),
        )

        return fig

    def _add_emotion_wheel_background(self, fig):
        """Add the emotion wheel background to the figure."""
        # Add emotion sectors
        for i, emotion in enumerate(CORE_EMOTIONS):
            angle = i * 45  # 360 / 8 = 45 degrees per emotion

            # Add sector labels
            fig.add_trace(go.Scatterpolar(
                r=[3.5],
                theta=[angle],
                text=[emotion],
                mode='text',
                textfont=dict(color=EMOTION_COLORS[emotion], size=14),
                showlegend=False,
            ))

            # Add colored sectors (faint background)
            theta = np.linspace(angle - 22.5, angle + 22.5, 20)
            r = np.ones(20) * 3

            fig.add_trace(go.Scatterpolar(
                r=r,
                theta=theta,
                fill='toself',
                fillcolor=f'rgba{self._hex_to_rgba(EMOTION_COLORS[emotion], 0.1)}',
                line=dict(color=EMOTION_COLORS[emotion], width=1),
                showlegend=False,
            ))

    def _hex_to_rgba(self, hex_color, alpha=1.0):
        """Convert hex color to RGBA tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)

    def _create_emotion_contour(self, emotion_scores, source="user"):
        """
        Create a continuous emotion contour based on emotion scores.

        Args:
            emotion_scores (dict): Dictionary mapping emotions to scores.
            source (str): Either 'user' or 'assistant'.

        Returns:
            tuple: (r, theta, z) arrays for the contour.
        """
        # Create a grid in polar coordinates
        theta_grid = np.linspace(0, 2*np.pi, self.resolution)
        r_grid = np.linspace(0, 3, self.resolution)
        theta_mesh, r_mesh = np.meshgrid(theta_grid, r_grid)

        # Initialize the intensity grid
        z_grid = np.zeros((self.resolution, self.resolution))

        # For each emotion, add its contribution to the grid
        for i, emotion in enumerate(CORE_EMOTIONS):
            # Get the emotion score
            score = emotion_scores.get(emotion, 0)

            # Calculate the angle for this emotion
            emotion_angle = i * 45 * np.pi / 180  # Convert to radians

            # Calculate the contribution of this emotion to each point in the grid
            # using a radial basis function
            for j in range(self.resolution):
                for k in range(self.resolution):
                    # Calculate the angular distance
                    angle_diff = min(
                        abs(theta_mesh[j, k] - emotion_angle),
                        2*np.pi - abs(theta_mesh[j, k] - emotion_angle)
                    )

                    # Calculate the distance from the emotion's peak
                    distance = np.sqrt(
                        (r_mesh[j, k] * np.cos(theta_mesh[j, k]) - score * np.cos(emotion_angle))**2 +
                        (r_mesh[j, k] * np.sin(theta_mesh[j, k]) - score * np.sin(emotion_angle))**2
                    )

                    # Add the contribution using a Gaussian function
                    # The width of the Gaussian controls how much emotions blend
                    width = 1.0  # Adjust this to control blending
                    z_grid[j, k] += score * np.exp(-(distance**2) / (2 * width**2))

        # Normalize the grid
        if np.max(z_grid) > 0:
            z_grid = z_grid / np.max(z_grid) * 3

        # Convert to Cartesian for plotting
        x = r_mesh * np.cos(theta_mesh)
        y = r_mesh * np.sin(theta_mesh)

        # Set color based on source
        color = "rgba(0, 100, 255, 0.7)" if source == "user" else "rgba(255, 100, 0, 0.7)"

        return x, y, z_grid, color

    def update_emotions(self, user_emotions=None, assistant_emotions=None):
        """
        Update the target emotion states.

        Args:
            user_emotions (dict, optional): New user emotion scores.
            assistant_emotions (dict, optional): New assistant emotion scores.
        """
        if user_emotions:
            self.target_user_emotions = user_emotions.copy()

        if assistant_emotions:
            self.target_assistant_emotions = assistant_emotions.copy()

    def _interpolate_emotions(self):
        """Interpolate between current and target emotion states for smooth transitions."""
        # Calculate time-based interpolation factor
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Calculate interpolation factor (ease-in-out)
        alpha = min(1.0, dt / self.transition_speed)

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

    def get_updated_figure(self):
        """
        Get the updated figure with the current emotion states.

        Returns:
            plotly.graph_objects.Figure: The updated figure.
        """
        # Interpolate emotions for smooth transitions
        self._interpolate_emotions()

        # Create a new figure with the base layout
        fig = self._create_base_figure()

        # Create a smoother visualization using more points for the polygons
        # This creates a more continuous representation of the emotional space

        # Generate smooth polygon points for user emotions (24 points instead of 8)
        user_r, user_theta = self._generate_smooth_polygon(self.current_user_emotions)

        # Add user emotion polygon with the smooth contour
        fig.add_trace(go.Scatterpolar(
            r=user_r,
            theta=user_theta,
            fill='toself',
            fillcolor='rgba(0,100,255,0.3)',
            line=dict(color='rgba(0,100,255,0.8)', width=2),
            name="User Emotions"
        ))

        # Generate smooth polygon points for assistant emotions
        assistant_r, assistant_theta = self._generate_smooth_polygon(self.current_assistant_emotions)

        # Add assistant emotion polygon with the smooth contour
        fig.add_trace(go.Scatterpolar(
            r=assistant_r,
            theta=assistant_theta,
            fill='toself',
            fillcolor='rgba(255,100,0,0.3)',
            line=dict(color='rgba(255,100,0,0.8)', width=2),
            name="Assistant Emotions"
        ))

        # Add markers for the peak emotions
        for i, emotion in enumerate(CORE_EMOTIONS):
            angle_deg = i * 45

            # User peak
            user_score = self.current_user_emotions[emotion]
            if user_score > 0.5:
                fig.add_trace(go.Scatterpolar(
                    r=[user_score],
                    theta=[angle_deg],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='blue',
                        symbol='circle',
                    ),
                    showlegend=False,
                    name=f"User {emotion}"
                ))

            # Assistant peak
            assistant_score = self.current_assistant_emotions[emotion]
            if assistant_score > 0.5:
                fig.add_trace(go.Scatterpolar(
                    r=[assistant_score],
                    theta=[angle_deg],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='orange',
                        symbol='circle',
                    ),
                    showlegend=False,
                    name=f"Assistant {emotion}"
                ))

        return fig

    def _generate_smooth_polygon(self, emotion_scores, num_points=24):
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

        # Interpolate x and y separately using cubic spline
        cs_x = CubicSpline(t, x, bc_type='periodic')
        cs_y = CubicSpline(t, y, bc_type='periodic')

        # Generate interpolated points
        x_interp = cs_x(t_interp)
        y_interp = cs_y(t_interp)

        # Convert back to polar coordinates
        r_interp = np.sqrt(x_interp**2 + y_interp**2)
        theta_interp = np.degrees(np.arctan2(y_interp, x_interp)) % 360

        return r_interp, theta_interp