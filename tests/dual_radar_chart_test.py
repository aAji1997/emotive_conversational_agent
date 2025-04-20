import dash
import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, callback
import random
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

# Helper function to format data for RadarChart
def format_data_for_radar(user_scores, assistant_scores=None):
    """Converts emotion score dictionaries to format needed for radar chart.

    Args:
        user_scores: Dictionary mapping emotions to scores for the user
        assistant_scores: Optional dictionary mapping emotions to scores for the assistant

    Returns:
        List of dictionaries with format: [{"emotion": emotion, "User": user_score, "Assistant": assistant_score}]
    """
    # Ensure all core emotions are present, defaulting to 0 if missing
    formatted = []
    for emotion in CORE_EMOTIONS:
        entry = {"emotion": emotion, "User": user_scores.get(emotion, 0)}

        # Add assistant scores if provided
        if assistant_scores is not None:
            entry["Assistant"] = assistant_scores.get(emotion, 0)

        formatted.append(entry)
    return formatted

# Generate random emotion scores for testing
def generate_random_scores():
    return {emotion: random.randint(0, 3) for emotion in CORE_EMOTIONS}

# Initial sample data
initial_user_scores = generate_random_scores()
initial_assistant_scores = generate_random_scores()
initial_radar_data = format_data_for_radar(initial_user_scores, initial_assistant_scores)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    withGlobalClasses=True,
    children=[
        html.Div([
            dmc.Title("Dual Emotion Radar Chart Test", order=2, ta="center"),
            dmc.Text("Displaying both user and assistant sentiment contours", ta="center", size="sm", c="dimmed"),
            dmc.Space(h=20),
            dmc.RadarChart(
                id="emotion-radar-chart",
                h=450, # Height of the chart
                data=initial_radar_data,
                dataKey="emotion", # Key in data dictionaries for axis labels
                withPolarGrid=True,
                withPolarAngleAxis=True,
                withPolarRadiusAxis=True,
                polarRadiusAxisProps={"domain": [0, 3], "tickCount": 4},
                series=[
                    # Define the data series to plot with contrasting colors and transparency
                    {"name": "User", "color": "blue.6", "opacity": 0.7},
                    {"name": "Assistant", "color": "red.6", "opacity": 0.7}
                ],
            ),
            dmc.Space(h=20),
            html.Div(id="update-log", style={"maxHeight": "200px", "overflow": "auto", "border": "1px solid #ddd", "padding": "10px"}),
            dcc.Interval(
                id='interval-component',
                interval=2000, # Update every 2 seconds (2000 milliseconds)
                n_intervals=0
            )
        ], style={"maxWidth": "800px", "margin": "auto", "padding": "20px"})
    ]
)

# Define the callback to update the chart
@app.callback(
    [Output('emotion-radar-chart', 'data'),
     Output('update-log', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_radar_chart(n):
    """Generates new random scores and updates the chart data."""
    # Generate new random scores for each emotion
    user_scores = generate_random_scores()
    assistant_scores = generate_random_scores()

    # Format the data for the radar chart
    new_radar_data = format_data_for_radar(user_scores, assistant_scores)

    # Create a log message
    timestamp = time.strftime("%H:%M:%S")
    log_message = f"[{timestamp}] Update #{n}: Chart updated with new user and assistant sentiment data"
    logger.info(log_message)

    # Return both the new data and the updated log
    return new_radar_data, html.Div([
        html.Div(log_message),
        html.Br(),
        html.Div(f"User scores: {user_scores}"),
        html.Div(f"Assistant scores: {assistant_scores}"),
        html.Hr(),
    ], style={"marginBottom": "10px"})

# Run the app
if __name__ == '__main__':
    print("Starting Dual Radar Chart Test...")
    print("This test demonstrates displaying both user and assistant sentiment contours")
    print("with contrasting colors and transparency.")
    print("\nAccess the chart at: http://127.0.0.1:8050/")
    app.run(debug=True, host='127.0.0.1', port=8050)
