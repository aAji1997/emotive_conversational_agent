"""
Memory Visualization Dashboard.

This module provides a Dash-based dashboard for visualizing the traced
interactions between memory components.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import Dash components
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_mantine_components as dmc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import memory tracing
from memory.memory_tracing import initialize_tracing, end_tracing

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if wandb and weave are available
try:
    import wandb
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    logger.warning("wandb or weave not installed. Visualization will have limited functionality.")

class MemoryVisualizationDashboard:
    """
    A Dash-based dashboard for visualizing memory component interactions.
    """
    
    def __init__(self, trace_data_path: Optional[str] = None):
        """
        Initialize the memory visualization dashboard.
        
        Args:
            trace_data_path: Optional path to a JSON file containing trace data
        """
        self.app = dash.Dash(__name__, title="Memory Component Visualization")
        self.trace_data_path = trace_data_path
        self.trace_data = self._load_trace_data()
        
        # Initialize the dashboard layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        
        logger.info("Memory visualization dashboard initialized")
    
    def _load_trace_data(self) -> List[Dict[str, Any]]:
        """Load trace data from a file or Weave."""
        if self.trace_data_path and os.path.exists(self.trace_data_path):
            try:
                with open(self.trace_data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading trace data from {self.trace_data_path}: {e}")
        
        # If no file or error loading, return empty list
        return []
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = dmc.MantineProvider(
            theme={"colorScheme": "light"},
            withGlobalClasses=True,
            children=[
                dmc.Container([
                    dmc.Title("Memory Component Interaction Visualization", order=1, align="center", mb=20),
                    
                    # Controls section
                    dmc.Paper([
                        dmc.Group([
                            dmc.Button(
                                "Refresh Data",
                                id="refresh-button",
                                variant="filled",
                                color="blue"
                            ),
                            dmc.Select(
                                id="visualization-type",
                                label="Visualization Type",
                                value="component-flow",
                                data=[
                                    {"value": "component-flow", "label": "Component Flow"},
                                    {"value": "timeline", "label": "Timeline"},
                                    {"value": "interaction-graph", "label": "Interaction Graph"}
                                ],
                                style={"width": 200}
                            ),
                            dmc.Switch(
                                id="auto-refresh",
                                label="Auto Refresh",
                                checked=False
                            )
                        ], position="apart"),
                    ], p=15, mb=20),
                    
                    # Main visualization area
                    dmc.Paper([
                        dcc.Loading(
                            id="loading-visualization",
                            type="circle",
                            children=[
                                html.Div(id="visualization-container", style={"height": "600px"})
                            ]
                        )
                    ], p=15, mb=20),
                    
                    # Trace details section
                    dmc.Paper([
                        dmc.Title("Trace Details", order=3, mb=10),
                        html.Div(id="trace-details", style={"maxHeight": "200px", "overflow": "auto"})
                    ], p=15),
                    
                    # Hidden components for state management
                    dcc.Store(id="trace-data-store"),
                    dcc.Interval(
                        id="auto-refresh-interval",
                        interval=5000,  # 5 seconds
                        disabled=True
                    )
                ], fluid=True, py=20)
            ]
        )
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        # Callback to toggle auto-refresh
        @self.app.callback(
            Output("auto-refresh-interval", "disabled"),
            Input("auto-refresh", "checked")
        )
        def toggle_auto_refresh(checked):
            return not checked
        
        # Callback to refresh data
        @self.app.callback(
            Output("trace-data-store", "data"),
            [Input("refresh-button", "n_clicks"),
             Input("auto-refresh-interval", "n_intervals")],
            prevent_initial_call=True
        )
        def refresh_trace_data(n_clicks, n_intervals):
            # Load the latest trace data
            if WEAVE_AVAILABLE:
                try:
                    # Get traces from Weave
                    traces = weave.get_traces()
                    return traces
                except Exception as e:
                    logger.error(f"Error getting traces from Weave: {e}")
            
            # Fall back to file data if available
            return self._load_trace_data()
        
        # Callback to update visualization
        @self.app.callback(
            Output("visualization-container", "children"),
            [Input("trace-data-store", "data"),
             Input("visualization-type", "value")]
        )
        def update_visualization(trace_data, viz_type):
            if not trace_data:
                return html.Div("No trace data available. Click 'Refresh Data' to load data.", 
                               style={"textAlign": "center", "marginTop": "50px"})
            
            if viz_type == "component-flow":
                return self._create_component_flow_viz(trace_data)
            elif viz_type == "timeline":
                return self._create_timeline_viz(trace_data)
            elif viz_type == "interaction-graph":
                return self._create_interaction_graph_viz(trace_data)
            else:
                return html.Div(f"Unknown visualization type: {viz_type}")
        
        # Callback to update trace details
        @self.app.callback(
            Output("trace-details", "children"),
            [Input("trace-data-store", "data")]
        )
        def update_trace_details(trace_data):
            if not trace_data:
                return "No trace data available"
            
            # Create a summary of the trace data
            return html.Pre(
                json.dumps(trace_data, indent=2),
                style={"whiteSpace": "pre-wrap", "fontSize": "12px"}
            )
    
    def _create_component_flow_viz(self, trace_data):
        """Create a component flow visualization."""
        # This is a placeholder for the actual implementation
        # In a real implementation, this would create a Sankey diagram or similar
        
        # For now, just return a message
        return html.Div([
            html.H3("Component Flow Visualization"),
            html.P("This would show a Sankey diagram of component interactions."),
            html.Pre(json.dumps(trace_data[:3], indent=2))
        ])
    
    def _create_timeline_viz(self, trace_data):
        """Create a timeline visualization."""
        # This is a placeholder for the actual implementation
        # In a real implementation, this would create a timeline chart
        
        # For now, just return a message
        return html.Div([
            html.H3("Timeline Visualization"),
            html.P("This would show a timeline of component interactions."),
            html.Pre(json.dumps(trace_data[:3], indent=2))
        ])
    
    def _create_interaction_graph_viz(self, trace_data):
        """Create an interaction graph visualization."""
        # This is a placeholder for the actual implementation
        # In a real implementation, this would create a network graph
        
        # For now, just return a message
        return html.Div([
            html.H3("Interaction Graph Visualization"),
            html.P("This would show a network graph of component interactions."),
            html.Pre(json.dumps(trace_data[:3], indent=2))
        ])
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        logger.info(f"Starting memory visualization dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)

def main():
    """Main entry point for the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Component Visualization Dashboard")
    parser.add_argument("--trace-data", type=str, help="Path to trace data JSON file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    dashboard = MemoryVisualizationDashboard(trace_data_path=args.trace_data)
    dashboard.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()
