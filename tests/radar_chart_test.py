import dash
import dash_mantine_components as dmc
# import dash_core_components as dcc # Deprecated
from dash import dcc # Use this import instead
from dash import html, Input, Output
import random

# Define the 8 core emotions based on Plutchik's wheel
CORE_EMOTIONS = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

# Helper function to format data for RadarChart
def format_data_for_radar(emotion_scores):
    """Converts {'Emotion': score} dict to [{'emotion': Emotion, 'Sentiment': score}] list."""
    return [
        {"emotion": emotion, "Sentiment": score}
        for emotion, score in emotion_scores.items()
    ]

# Initial sample data (can be replaced by callback generation)
initial_scores = {emotion: random.randint(0, 3) for emotion in CORE_EMOTIONS}
initial_radar_data = format_data_for_radar(initial_scores)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = dmc.MantineProvider(
    theme={"colorScheme": "light"},
    withGlobalClasses=True, # Renamed from withGlobalStyles
    # withNormalizeCSS=True, # Removed, normalization is often default now
    children=[
        html.Div([
            dmc.Title("Real-time Emotion Radar Chart", order=2),
            dmc.RadarChart(
                id="emotion-radar-chart",
                h=400, # Height of the chart
                data=initial_radar_data,
                dataKey="emotion", # Key in data dictionaries for axis labels
                withPolarGrid=True,
                withPolarAngleAxis=True,
                withPolarRadiusAxis=True,
                series=[
                    # Define the data series to plot
                    {"name": "Sentiment", "color": "blue.6", "opacity": 0.7}
                ],
            ),
            dcc.Interval(
                id='interval-component',
                interval=1*1000, # Update every 1 second (1000 milliseconds)
                n_intervals=0
            )
        ])
    ]
)

# Define the callback to update the chart
@app.callback(
    Output('emotion-radar-chart', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_radar_chart(n):
    """Generates new random scores and updates the chart data."""
    # Generate new random scores for each emotion
    new_scores = {emotion: random.randint(0, 3) for emotion in CORE_EMOTIONS}
    # Format the data for the radar chart
    new_radar_data = format_data_for_radar(new_scores)
    return new_radar_data

# Run the app
if __name__ == '__main__':
    # app.run_server(debug=True) # Obsolete
    app.run(debug=True) # Use app.run instead 