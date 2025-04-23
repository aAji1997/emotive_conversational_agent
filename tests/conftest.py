"""
Configure test environment.
This file is automatically loaded by pytest.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
