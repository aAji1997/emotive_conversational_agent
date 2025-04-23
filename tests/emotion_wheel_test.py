"""
Test implementation for the Emotion Wheel Visualization.

This script demonstrates the continuous emotional space visualization
with smooth transitions between emotional states.
"""

import sys
import os
import time
import random
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the emotion wheel visualization
from gpt_realtime.emotion_wheel_visualization import EmotionWheelVisualization
from gpt_realtime.emotion_transition_manager import EmotionTransitionManager

# Import Dash components
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

# Create the Dash app
app = dash.Dash(__name__)

# Create the emotion wheel visualization
emotion_wheel = EmotionWheelVisualization(resolution=50, transition_speed=0.5)

# Create the emotion transition manager
emotion_manager = EmotionTransitionManager(transition_speed=0.5, history_length=20)

# Generate random emotion scores for testing
def generate_random_scores():
    """Generate random emotion scores with some coherence."""
    # Start with previous target emotions to create coherence
    base_emotions = emotion_manager.target_user_emotions.copy()

    # Add random changes
    for emotion in CORE_EMOTIONS:
        # Random change with some constraints to create more natural transitions
        change = random.uniform(-0.5, 0.5)
        base_emotions[emotion] = max(0, min(3, base_emotions[emotion] + change))

    return base_emotions

# Create hidden divs for slider callbacks
hidden_divs = []
for emotion in CORE_EMOTIONS:
    hidden_divs.append(html.Div(id=f'user-{emotion.lower()}-output', style={'display': 'none'}))
    hidden_divs.append(html.Div(id=f'assistant-{emotion.lower()}-output', style={'display': 'none'}))

# Define the layout
app.layout = html.Div([
    html.H1("Emotional GPS - Continuous Visualization", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            dcc.Graph(
                id='emotion-wheel-graph',
                figure=emotion_wheel.get_updated_figure(),
                config={'displayModeBar': False}
            ),
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.H3("Current Emotion Scores"),
            html.Div(id='user-emotions-display'),
            html.Hr(),
            html.Div(id='assistant-emotions-display'),
            html.Hr(),
            html.Button('Generate Random Emotions', id='generate-button', n_clicks=0),
            html.Div(id='generation-info', style={'marginTop': '10px'}),

            # Sliders for manual control
            html.H3("Manual Control (User Emotions)", style={'marginTop': '20px'}),
            *[
                html.Div([
                    html.Label(emotion),
                    dcc.Slider(
                        id=f'user-{emotion.lower()}-slider',
                        min=0,
                        max=3,
                        step=0.1,
                        value=0,
                        marks={i: str(i) for i in range(4)},
                    )
                ]) for emotion in CORE_EMOTIONS
            ],

            html.H3("Manual Control (Assistant Emotions)", style={'marginTop': '20px'}),
            *[
                html.Div([
                    html.Label(emotion),
                    dcc.Slider(
                        id=f'assistant-{emotion.lower()}-slider',
                        min=0,
                        max=3,
                        step=0.1,
                        value=0,
                        marks={i: str(i) for i in range(4)},
                    )
                ]) for emotion in CORE_EMOTIONS
            ],

        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    ]),

    # Add all the hidden divs for slider callbacks
    *hidden_divs,

    # Interval component for updating the visualization
    dcc.Interval(
        id='interval-component',
        interval=50,  # Update every 50 milliseconds for smooth animation
        n_intervals=0
    )
])

# Callback to update the visualization
@app.callback(
    Output('emotion-wheel-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_emotion_wheel(_):
    """Update the emotion wheel visualization."""
    return emotion_wheel.get_updated_figure()

# Callback to update the emotion displays
@app.callback(
    [Output('user-emotions-display', 'children'),
     Output('assistant-emotions-display', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_emotion_displays(_):
    """Update the emotion displays with current values."""
    user_emotions, assistant_emotions = emotion_manager.get_current_emotions()

    user_display = [
        html.H4("User Emotions"),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Emotion"), html.Th("Score")])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(emotion),
                    html.Td(f"{score:.2f}")
                ]) for emotion, score in user_emotions.items()
            ])
        ])
    ]

    assistant_display = [
        html.H4("Assistant Emotions"),
        html.Table([
            html.Thead([
                html.Tr([html.Th("Emotion"), html.Th("Score")])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(emotion),
                    html.Td(f"{score:.2f}")
                ]) for emotion, score in assistant_emotions.items()
            ])
        ])
    ]

    return user_display, assistant_display

# Callback for the generate button
@app.callback(
    Output('generation-info', 'children'),
    Input('generate-button', 'n_clicks'),
    prevent_initial_call=True
)
def generate_random_emotions(n_clicks):
    """Generate random emotions when the button is clicked."""
    if n_clicks > 0:
        # Generate random emotions
        user_emotions = generate_random_scores()
        assistant_emotions = generate_random_scores()

        # Update the emotion manager
        emotion_manager.update_emotions(user_emotions, assistant_emotions)

        # Update the emotion wheel
        emotion_wheel.update_emotions(user_emotions, assistant_emotions)

        return f"Generated new random emotions (#{n_clicks})"

    return ""

# Create callbacks for user emotion sliders
for emotion in CORE_EMOTIONS:
    @app.callback(
        Output(f'user-{emotion.lower()}-output', 'children'),
        Input(f'user-{emotion.lower()}-slider', 'value'),
        prevent_initial_call=True
    )
    def update_user_emotion(value, emotion=emotion):
        """Update user emotions when sliders are moved."""
        # Get current emotions
        user_emotions, _ = emotion_manager.get_current_emotions()

        # Update the specific emotion
        user_emotions[emotion] = value

        # Update the emotion manager and wheel
        emotion_manager.update_emotions(user_emotions=user_emotions)
        emotion_wheel.update_emotions(user_emotions=user_emotions)

        return f"Updated {emotion} to {value}"

# Create callbacks for assistant emotion sliders
for emotion in CORE_EMOTIONS:
    @app.callback(
        Output(f'assistant-{emotion.lower()}-output', 'children'),
        Input(f'assistant-{emotion.lower()}-slider', 'value'),
        prevent_initial_call=True
    )
    def update_assistant_emotion(value, emotion=emotion):
        """Update assistant emotions when sliders are moved."""
        # Get current emotions
        _, assistant_emotions = emotion_manager.get_current_emotions()

        # Update the specific emotion
        assistant_emotions[emotion] = value

        # Update the emotion manager and wheel
        emotion_manager.update_emotions(assistant_emotions=assistant_emotions)
        emotion_wheel.update_emotions(assistant_emotions=assistant_emotions)

        return f"Updated {emotion} to {value}"

# Run the app
if __name__ == '__main__':
    print("Starting Emotion Wheel Test...")
    print("This test demonstrates a continuous emotional space visualization")
    print("with smooth transitions between emotional states.")
    print("\nControls:")
    print("- Use the sliders to manually adjust emotion values")
    print("- Click 'Generate Random Emotions' to generate random emotion states")
    print("\nThe visualization will update automatically with smooth transitions.")

    app.run(debug=True, host='127.0.0.1')
