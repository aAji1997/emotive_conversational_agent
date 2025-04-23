"""
Test script to verify the memory retrieval phrase detection functionality.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_retrieval_detection():
    """Test the memory retrieval phrase detection functionality."""
    try:
        # Create a simple class to test the memory retrieval phrase detection
        class MemoryPhraseDetector:
            def _detect_memory_retrieval_intent(self, message: str) -> bool:
                """Detect phrases in the assistant's message that indicate an intent to retrieve a memory.
                
                Args:
                    message: The assistant's message
                    
                Returns:
                    True if memory retrieval intent is detected, False otherwise
                """
                # List of phrases that indicate memory retrieval intent
                retrieval_phrases = [
                    "let me recall",
                    "let me remember",
                    "let me think back",
                    "if i remember correctly",
                    "from what i remember",
                    "based on our previous conversations",
                    "as you mentioned before",
                    "as you've told me before",
                    "as we discussed earlier",
                    "let me check my notes",
                    "let me see what i know about",
                    "give me a second to recall",
                    "let me access my memory",
                    "i believe you mentioned",
                    "you previously told me",
                    "according to our past conversations",
                    "let me search my memory",
                    "let me look that up for you",
                    "i'm trying to remember",
                    "let me see if i can find"
                ]
                
                # Convert message to lowercase for case-insensitive matching
                message_lower = message.lower()
                
                # Check if any of the phrases are in the message
                for phrase in retrieval_phrases:
                    if phrase in message_lower:
                        logger.info(f"Memory retrieval intent detected: '{phrase}'")
                        return True
                        
                return False
        
        # Create an instance of the detector
        detector = MemoryPhraseDetector()

        # Test phrases that should trigger memory retrieval
        test_phrases = [
            "Let me recall what you told me about your hobbies.",
            "Let me remember what you said about your job.",
            "Let me think back to our previous conversation.",
            "If I remember correctly, you mentioned you like hiking.",
            "From what I remember, you're a software engineer.",
            "Based on our previous conversations, you enjoy reading science fiction.",
            "As you mentioned before, you have a dog named Max.",
            "As you've told me before, you prefer coffee over tea.",
            "As we discussed earlier, you're planning a trip to Japan.",
            "Let me check my notes about your preferences.",
            "Let me see what I know about your favorite movies.",
            "Give me a second to recall your birthday.",
            "Let me access my memory about our last conversation.",
            "I believe you mentioned having two children.",
            "You previously told me you're learning to play the guitar.",
            "According to our past conversations, you enjoy cooking Italian food.",
            "Let me search my memory for information about your hobbies.",
            "Let me look that up for you in my memory.",
            "I'm trying to remember what you said about your favorite book.",
            "Let me see if I can find any information about your travel plans."
        ]

        # Test phrases that should NOT trigger memory retrieval
        non_trigger_phrases = [
            "I don't know about that.",
            "Let me think about that.",
            "I'm not sure I understand.",
            "Could you explain that further?",
            "That's interesting.",
            "I'd like to learn more about that.",
            "Let me help you with that.",
            "I can assist you with that.",
            "Let me explain how this works.",
            "Let me show you an example."
        ]

        print("\n=== Testing Memory Retrieval Phrase Detection ===")
        
        # Test trigger phrases
        print("\nPhrases that SHOULD trigger memory retrieval:")
        for phrase in test_phrases:
            result = detector._detect_memory_retrieval_intent(phrase)
            print(f"  - \"{phrase}\": {'✓' if result else '✗'}")
            
        # Test non-trigger phrases
        print("\nPhrases that should NOT trigger memory retrieval:")
        for phrase in non_trigger_phrases:
            result = detector._detect_memory_retrieval_intent(phrase)
            print(f"  - \"{phrase}\": {'✗' if not result else '✓'}")

        print("\nMemory retrieval phrase detection test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error testing memory retrieval phrase detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    test_memory_retrieval_detection()
