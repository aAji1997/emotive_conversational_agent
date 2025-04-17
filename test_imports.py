"""
Test script to verify import paths.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Try importing from memory package
try:
    from memory.memory_utils import MemoryManager
    print("Successfully imported MemoryManager from memory.memory_utils")
except ImportError as e:
    print(f"Error importing MemoryManager: {e}")

# Try importing from memory_agent
try:
    from memory.memory_agent import ConversationMemoryAgent
    print("Successfully imported ConversationMemoryAgent from memory.memory_agent")
except ImportError as e:
    print(f"Error importing ConversationMemoryAgent: {e}")

# Print the Python path for debugging
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")
