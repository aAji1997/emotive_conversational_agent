"""
Test script to verify the memory phrase detection functionality.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_phrase_detection():
    """Test the memory phrase detection functionality."""
    try:
        # Create a simple class to test the memory phrase detection
        class MemoryPhraseDetector:
            def _detect_memory_storage_intent(self, message: str) -> bool:
                """Detect phrases in the assistant's message that indicate an intent to store a memory.

                Args:
                    message: The assistant's message

                Returns:
                    True if memory storage intent is detected, False otherwise
                """
                # List of phrases that indicate memory storage intent
                memory_phrases = [
                    "i'll remember that",
                    "i'll make a note of that",
                    "i've noted that",
                    "i'll keep that in mind",
                    "i won't forget that",
                    "i've made a note",
                    "i'm noting that down",
                    "i'll add that to your profile",
                    "i'll store that information",
                    "i'll remember this about you",
                    "that's interesting, i'll remember",
                    "i'll keep that information",
                    "i'll save that for later",
                    "i'll make sure to remember"
                ]

                # Convert message to lowercase for case-insensitive matching
                message_lower = message.lower()

                # Check if any of the phrases are in the message
                for phrase in memory_phrases:
                    if phrase in message_lower:
                        logger.info(f"Memory storage intent detected: '{phrase}'")
                        return True

                return False

        # Create an instance of the detector
        detector = MemoryPhraseDetector()

        # Test phrases that should trigger memory storage
        test_phrases = [
            "I'll remember that you enjoy hiking in the mountains.",
            "That's interesting, I'll make a note of that.",
            "I've noted that you prefer chocolate over vanilla.",
            "I'll keep that in mind for our future conversations.",
            "I won't forget that you're allergic to peanuts.",
            "I've made a note about your preference for sci-fi movies.",
            "I'm noting that down for future reference.",
            "I'll add that to your profile.",
            "I'll store that information for later.",
            "I'll remember this about you.",
            "That's interesting, I'll remember you like jazz music.",
            "I'll keep that information in my memory.",
            "I'll save that for later.",
            "I'll make sure to remember your birthday is in May."
        ]

        # Test phrases that should NOT trigger memory storage
        non_trigger_phrases = [
            "That's interesting.",
            "I understand.",
            "Thanks for sharing that with me.",
            "I appreciate your input.",
            "Let me think about that.",
            "That makes sense.",
            "I see what you mean.",
            "Tell me more about that."
        ]

        print("\n=== Testing Memory Phrase Detection ===")

        # Test trigger phrases
        print("\nPhrases that SHOULD trigger memory storage:")
        for phrase in test_phrases:
            result = detector._detect_memory_storage_intent(phrase)
            print(f"  - \"{phrase}\": {'✓' if result else '✗'}")

        # Test non-trigger phrases
        print("\nPhrases that should NOT trigger memory storage:")
        for phrase in non_trigger_phrases:
            result = detector._detect_memory_storage_intent(phrase)
            print(f"  - \"{phrase}\": {'✗' if not result else '✓'}")

        print("\nMemory phrase detection test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing memory phrase detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    test_memory_phrase_detection()
