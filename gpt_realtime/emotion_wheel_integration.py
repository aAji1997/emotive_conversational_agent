"""
Emotion Wheel Integration for the main application.

This module provides integration between the EmotionWheelVisualization and the main application.
It creates a Dash component that can be used in the main application layout.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import time

from gpt_realtime.emotion_wheel_visualization import EmotionWheelVisualization
from gpt_realtime.shared_emotion_state import SharedEmotionState, CORE_EMOTIONS

class EmotionWheelComponent:
    """
    A class for integrating the EmotionWheelVisualization with the main application.
    """
    
    def __init__(self, app, shared_emotion_scores=None):
        """
        Initialize the emotion wheel component.
        
        Args:
            app: The Dash application instance.
            shared_emotion_scores: A SharedEmotionState instance for accessing emotion scores.
        """
        self.app = app
        self.shared_emotion_scores = shared_emotion_scores
        
        # Create the emotion wheel visualization
        self.emotion_wheel = EmotionWheelVisualization(resolution=100, transition_speed=0.3)
        
        # Last update time for controlling update frequency
        self.last_update_time = time.time()
        self.last_data = None
        
        # Set up the callback for updating the visualization
        self._setup_callback()
    
    def get_component(self):
        """
        Get the Dash component for the emotion wheel visualization.
        
        Returns:
            dash.html.Div: The Dash component for the emotion wheel visualization.
        """
        return html.Div([
            dcc.Graph(
                id='emotion-wheel-graph',
                figure=self.emotion_wheel.get_updated_figure(),
                config={'displayModeBar': False},
                style={'height': '450px'}
            ),
            dcc.Interval(
                id='emotion-wheel-interval',
                interval=500,  # Update every 0.5 seconds for smoother updates
                n_intervals=0,
                max_intervals=-1  # Run indefinitely
            )
        ])
    
    def _setup_callback(self):
        """Set up the callback for updating the visualization."""
        @self.app.callback(
            Output('emotion-wheel-graph', 'figure'),
            Input('emotion-wheel-interval', 'n_intervals')
        )
        def update_emotion_wheel(_):
            """Update the emotion wheel visualization."""
            # Allow more frequent updates for smoother animation
            current_time = time.time()
            if current_time - self.last_update_time < 0.3:  # Only update once every 0.3 seconds at most
                if self.last_data is not None:
                    return self.last_data
            
            # Update the last update time
            self.last_update_time = current_time
            
            try:
                # Get emotion scores from the shared state
                if self.shared_emotion_scores is not None:
                    # If it's a SharedEmotionState instance
                    if hasattr(self.shared_emotion_scores, 'get_emotion_scores'):
                        user_scores, assistant_scores = self.shared_emotion_scores.get_emotion_scores()
                        
                        # Update the emotion wheel with the new scores
                        self.emotion_wheel.update_emotions(
                            user_emotions=user_scores,
                            assistant_emotions=assistant_scores
                        )
                
                # Get the updated figure
                figure = self.emotion_wheel.get_updated_figure()
                self.last_data = figure
                return figure
            
            except Exception as e:
                print(f"Error updating emotion wheel: {e}")
                # Return the last known good figure or a default figure
                if self.last_data is not None:
                    return self.last_data
                return self.emotion_wheel.get_updated_figure()
