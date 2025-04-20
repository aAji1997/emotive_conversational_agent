#!/usr/bin/env python3
"""
Run the memory visualization dashboard.

This script runs the memory visualization dashboard to display
traced interactions between memory components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import the memory visualization dashboard
from memory.memory_visualization import MemoryVisualizationDashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run the memory visualization dashboard")
    parser.add_argument("--trace-data", type=str, help="Path to trace data JSON file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Create and run the dashboard
    dashboard = MemoryVisualizationDashboard(trace_data_path=args.trace_data)
    
    print(f"\n{'='*80}")
    print(f"Memory Visualization Dashboard")
    print(f"{'='*80}")
    print(f"Access the dashboard at: http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop the dashboard")
    print(f"{'='*80}\n")
    
    dashboard.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()
