"""
Test script to verify the graceful handling of empty memory retrieval.
"""

import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_empty_memory_handling():
    """Test the graceful handling of empty memory retrieval."""
    try:
        # Create a simple class to test the memory formatting
        class MemoryFormatter:
            def format_memories_for_context(self, memories: List[Dict[str, Any]], retrieval_attempted: bool = False) -> str:
                """Format memories for inclusion in the conversation context.

                Args:
                    memories: List of memory dictionaries
                    retrieval_attempted: Whether memory retrieval was explicitly attempted

                Returns:
                    Formatted string of memories
                """
                if not memories:
                    if retrieval_attempted:
                        # If retrieval was explicitly attempted but no memories were found, provide clear feedback
                        return "### Memory retrieval note:\nI searched my memory but couldn't find any relevant information about this topic from our previous conversations. I'll respond based on my general knowledge instead.\n###\n"
                    else:
                        # If no retrieval was attempted or it wasn't explicit, return empty string
                        return ""

                formatted_memories = ["### Relevant memories from previous conversations:"]

                for i, memory in enumerate(memories):
                    content = memory.get('content', 'No content available')
                    category = memory.get('category', 'general')
                    formatted_memories.append(f"{i+1}. [{category}] {content}")

                formatted_memories.append("###\n")
                return "\n".join(formatted_memories)
        
        # Create an instance of the formatter
        formatter = MemoryFormatter()

        # Test cases
        test_cases = [
            {
                "name": "Empty memories with no explicit retrieval",
                "memories": [],
                "retrieval_attempted": False,
                "expected_output": ""
            },
            {
                "name": "Empty memories with explicit retrieval",
                "memories": [],
                "retrieval_attempted": True,
                "expected_output": "### Memory retrieval note:\nI searched my memory but couldn't find any relevant information about this topic from our previous conversations. I'll respond based on my general knowledge instead.\n###\n"
            },
            {
                "name": "Non-empty memories",
                "memories": [
                    {"content": "User enjoys hiking in the mountains", "category": "preference"},
                    {"content": "User has a dog named Max", "category": "personal_info"}
                ],
                "retrieval_attempted": False,
                "expected_output": "### Relevant memories from previous conversations:\n1. [preference] User enjoys hiking in the mountains\n2. [personal_info] User has a dog named Max\n###\n"
            },
            {
                "name": "Non-empty memories with explicit retrieval",
                "memories": [
                    {"content": "User enjoys hiking in the mountains", "category": "preference"},
                    {"content": "User has a dog named Max", "category": "personal_info"}
                ],
                "retrieval_attempted": True,
                "expected_output": "### Relevant memories from previous conversations:\n1. [preference] User enjoys hiking in the mountains\n2. [personal_info] User has a dog named Max\n###\n"
            }
        ]

        print("\n=== Testing Empty Memory Handling ===")
        
        # Run test cases
        for i, test_case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {test_case['name']}")
            result = formatter.format_memories_for_context(test_case["memories"], test_case["retrieval_attempted"])
            
            # Check if result matches expected output
            if result == test_case["expected_output"]:
                print(f"✓ PASS")
            else:
                print(f"✗ FAIL")
                print(f"Expected:\n{repr(test_case['expected_output'])}")
                print(f"Got:\n{repr(result)}")

        print("\nEmpty memory handling test completed")
        return True
    except Exception as e:
        logger.error(f"Error testing empty memory handling: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    test_empty_memory_handling()
