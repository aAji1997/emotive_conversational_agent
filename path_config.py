"""
Configure Python path to include the project root directory.
Import this file at the beginning of any script that needs to access project modules.
"""

import os
import sys
from pathlib import Path

# Get the absolute path to the project root directory
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Print confirmation for debugging
if __name__ == "__main__":
    print(f"Added {project_root} to Python path")
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")
