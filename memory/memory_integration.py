"""
Memory integration module for the conversational agent.

This module provides functions to integrate the memory agent with both
real-time audio conversations and chat completions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.runners import Runner

from memory.memory_agent import ConversationMemoryAgent

# Set up logging
logger = logging.getLogger(__name__)

class MemoryIntegration:
    """Integration class for the memory agent."""

    def __init__(self, openai_api_key: str, gemini_api_key: str = None, user_id: str = None, supabase_url: str = None, supabase_key: str = None):
        """Initialize the memory integration.

        Args:
            openai_api_key: OpenAI API key for the LLM models
            gemini_api_key: Optional Gemini API key for alternative models
            user_id: Optional user ID for user-specific memories
            supabase_url: Optional Supabase URL for database connection
            supabase_key: Optional Supabase API key for database connection
        """
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.user_id = user_id
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key

        # Initialize session service
        self.session_service = InMemorySessionService()

        # Initialize models with cost-effective GPT-4o-mini
        self.storage_model = LiteLlm(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.2,  # Lower temperature for more consistent memory storage
            max_tokens=1024   # Limit token usage for efficiency
        )

        self.retrieval_model = LiteLlm(
            model="gpt-4o-mini",
            api_key=openai_api_key,
            temperature=0.3,  # Slightly higher temperature for retrieval
            max_tokens=1024   # Limit token usage for efficiency
        )

        # Use Gemini 2.0 Flash directly for the orchestrator model
        if gemini_api_key:
            # Set the API key in the environment for direct model usage
            import os
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
            # Use the direct model name for Gemini in ADK
            self.orchestrator_model = "gemini-2.0-flash"  # Direct model name for Gemini in ADK
        else:
            # Fall back to GPT-4o-mini if Gemini key is not available
            self.orchestrator_model = LiteLlm(
                model="gpt-4o-mini",
                api_key=openai_api_key,
                temperature=0.1
            )

        # Initialize the memory agent
        self.memory_agent = self._create_memory_agent()

        logger.info("Memory integration initialized with GPT-4o-mini for storage/retrieval and Gemini 2.0 Flash for orchestration")

    def _create_memory_agent(self) -> ConversationMemoryAgent:
        """Create a memory agent with the configured models.

        Returns:
            Configured ConversationMemoryAgent instance
        """
        # Create the memory agent
        memory_agent = ConversationMemoryAgent(
            storage_agent_model=self.storage_model,
            retrieval_agent_model=self.retrieval_model,
            orchestrator_agent_model=self.orchestrator_model,
            session_service=self.session_service,
            user_id=self.user_id,
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key
        )

        return memory_agent

    async def process_user_message(self, message: str, conversation_mode: str) -> None:
        """Process a user message and store it in memory if relevant.

        Args:
            message: The user's message
            conversation_mode: 'audio' or 'chat'
        """
        print(f"\n[MEMORY AGENT] Processing user message in {conversation_mode} mode: '{message[:50]}...'")
        logger.info(f"Processing user message in {conversation_mode} mode")
        await self.memory_agent.process_message(message, "user", conversation_mode)
        print(f"\n[MEMORY AGENT] Finished processing user message in {conversation_mode} mode")

    async def process_assistant_message(self, message: str, conversation_mode: str) -> None:
        """Process an assistant message and store it in memory if relevant.

        Args:
            message: The assistant's message
            conversation_mode: 'audio' or 'chat'
        """
        print(f"\n[MEMORY AGENT] Processing assistant message in {conversation_mode} mode: '{message[:50]}...'")
        logger.info(f"Processing assistant message in {conversation_mode} mode")

        # Check if the message contains phrases indicating memory storage or retrieval intent
        storage_intent = self._detect_memory_storage_intent(message)
        retrieval_intent = self._detect_memory_retrieval_intent(message)

        # Process the message with the memory agent
        if storage_intent:
            # If storage intent is detected, use a special flag to prioritize storage
            print(f"\n[MEMORY AGENT] Detected memory storage intent in assistant message")
            await self.memory_agent.process_message(message, "assistant", conversation_mode, force_memory_consideration=True)
            logger.info("Detected memory storage intent in assistant message")
        elif retrieval_intent:
            # If retrieval intent is detected, use a special flag to prioritize retrieval
            print(f"\n[MEMORY AGENT] Detected memory retrieval intent in assistant message")
            await self.memory_agent.process_message(message, "assistant", conversation_mode, force_memory_retrieval=True)
            logger.info("Detected memory retrieval intent in assistant message")
        else:
            # Normal processing
            print(f"\n[MEMORY AGENT] No specific intent detected, processing normally")
            await self.memory_agent.process_message(message, "assistant", conversation_mode)

        print(f"\n[MEMORY AGENT] Finished processing assistant message in {conversation_mode} mode")

    async def _centralized_memory_retrieval(self, current_message: str, force_retrieval: bool = False) -> List[Dict[str, Any]]:
        """Centralized function for retrieving memories to avoid redundant calls.

        Args:
            current_message: The current message to find relevant memories for
            force_retrieval: If True, prioritize memory retrieval

        Returns:
            List of relevant memory dictionaries
        """
        try:
            # Check if the message contains phrases indicating memory retrieval intent
            if not force_retrieval:
                force_retrieval = self._detect_memory_retrieval_intent(current_message)
                if force_retrieval:
                    logger.info("Detected memory retrieval intent in message")
                    print(f"\n[MEMORY INTEGRATION] Detected memory retrieval intent in message: '{current_message[:50]}...'")

            # For location-related queries, always force retrieval
            location_terms = ["where", "live", "location", "address", "city", "state", "country", "from", "reside"]
            if any(term in current_message.lower() for term in location_terms):
                force_retrieval = True
                print(f"\n[MEMORY INTEGRATION] Location-related query detected, forcing memory retrieval: '{current_message[:50]}...'")
                logger.info("Location-related query detected, forcing memory retrieval")

            # Create a dedicated session for this operation
            session = self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user")

            # Store the current message in the session for context
            session.state["current_message"] = current_message
            session.state["force_memory_retrieval"] = force_retrieval

            # Get relevant memories using the memory agent with optimized retrieval
            print(f"\n[MEMORY INTEGRATION] Getting relevant memories for: '{current_message[:50]}...'")
            relevant_memories = await self.memory_agent.get_relevant_memories(current_message, force_retrieval)

            logger.info(f"Retrieved {len(relevant_memories)} memories for context")

            # Print statement for visibility
            if relevant_memories:
                print(f"\n[MEMORY RETRIEVAL] Retrieved {len(relevant_memories)} memories for user {self.user_id or 'None'}:")
                for i, memory in enumerate(relevant_memories):
                    print(f"  {i+1}. {memory.get('content')} (Category: {memory.get('category', 'unknown')})")
            else:
                print(f"\n[MEMORY RETRIEVAL] No memories found for user {self.user_id or 'None'} matching: {current_message[:50]}...")

            return relevant_memories
        except Exception as e:
            logger.error(f"Error in centralized memory retrieval: {e}")
            print(f"\n[MEMORY INTEGRATION] Error in centralized memory retrieval: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return []  # Return empty list on error

    async def get_context_with_memories(self, current_message: str, force_retrieval: bool = False) -> str:
        """Get relevant memories for the current message and format them for context.

        Args:
            current_message: The current message to find relevant memories for
            force_retrieval: If True, prioritize memory retrieval

        Returns:
            Formatted string of memories for inclusion in the prompt
        """
        try:
            # Use the centralized memory retrieval function
            relevant_memories = await self._centralized_memory_retrieval(current_message, force_retrieval)

            # Format memories for context, indicating if retrieval was explicitly attempted
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories, retrieval_attempted=force_retrieval)

            # Print statement for visibility
            if relevant_memories:
                print(f"\n[MEMORY INTEGRATION] Memory context: '{memory_context[:100]}...'")
            else:
                print(f"\n[MEMORY INTEGRATION] No memory context to add")

            return memory_context
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            print(f"\n[MEMORY INTEGRATION] Error getting memory context: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return ""  # Return empty string on error

    async def enhance_prompt_with_memories(self, prompt: str, current_message: str) -> str:
        """Enhance a prompt with relevant memories.

        Args:
            prompt: The original prompt
            current_message: The current message to find relevant memories for

        Returns:
            Enhanced prompt with memories
        """
        try:
            # Use the centralized memory retrieval function
            force_retrieval = self._detect_memory_retrieval_intent(current_message)
            relevant_memories = await self._centralized_memory_retrieval(current_message, force_retrieval)

            # Format memories for context, indicating if retrieval was explicitly attempted
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories, retrieval_attempted=force_retrieval)

            if memory_context:
                # Insert memories before the prompt
                enhanced_prompt = f"{memory_context}\n\n{prompt}"
                logger.info("Enhanced prompt with memory context")
                return enhanced_prompt

            return prompt  # Return original prompt if no memories
        except Exception as e:
            logger.error(f"Error enhancing prompt with memories: {e}")
            return prompt  # Return original prompt on error
        finally:
            # Clean up the session
            # No need to explicitly close ADK sessions
            pass

    async def enhance_chat_messages(self, messages: List[Dict[str, str]], current_message: str) -> List[Dict[str, str]]:
        """Enhance chat messages with relevant memories.

        Args:
            messages: List of chat messages in the format [{"role": "...", "content": "..."}]
            current_message: The current message to find relevant memories for

        Returns:
            Enhanced messages with memories
        """
        try:
            # Use the centralized memory retrieval function
            force_retrieval = self._detect_memory_retrieval_intent(current_message)
            relevant_memories = await self._centralized_memory_retrieval(current_message, force_retrieval)

            # Print the retrieved memories for debugging
            print(f"\n[MEMORY INTEGRATION] Retrieved {len(relevant_memories)} memories for message: '{current_message[:50]}...'")
            for i, memory in enumerate(relevant_memories):
                print(f"  Memory {i+1}: {memory.get('content', 'No content')} (ID: {memory.get('id', 'unknown')})")

            # Format memories for context, indicating if retrieval was explicitly attempted
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories, retrieval_attempted=force_retrieval)

            # Print the formatted memory context for debugging
            print(f"\n[MEMORY INTEGRATION] Formatted memory context:\n{memory_context}")

            if not memory_context:
                print(f"\n[MEMORY INTEGRATION] No memory context to add to messages")
                return messages  # Return original messages if no memories

            # Create a new system message with memories or enhance existing one
            system_message_exists = False
            enhanced_messages = []

            for message in messages:
                if message["role"] == "system":
                    # Enhance existing system message
                    enhanced_content = f"{message['content']}\n\n{memory_context}"
                    message["content"] = enhanced_content
                    system_message_exists = True
                    print(f"\n[MEMORY INTEGRATION] Enhanced existing system message with memories")
                enhanced_messages.append(message)

            # Add a new system message with memories if none exists
            if not system_message_exists:
                memory_system_message = {
                    "role": "system",
                    "content": f"You are a helpful assistant with access to the user's previous conversations.\n\n{memory_context}"
                }
                enhanced_messages.insert(0, memory_system_message)
                print(f"\n[MEMORY INTEGRATION] Added new system message with memories")

            # Print the final enhanced messages for debugging
            print(f"\n[MEMORY INTEGRATION] Final enhanced messages:")
            for i, msg in enumerate(enhanced_messages):
                if msg["role"] == "system":
                    print(f"  System message: {msg['content'][:100]}...")

            logger.info("Enhanced chat messages with memory context")
            return enhanced_messages
        except Exception as e:
            logger.error(f"Error enhancing chat messages with memories: {e}")
            print(f"\n[MEMORY INTEGRATION] Error enhancing chat messages: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return messages  # Return original messages on error
        finally:
            # Clean up the session
            # No need to explicitly close ADK sessions
            pass

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
            "i'll make sure to remember",
            "i'll remember",
            "i will remember",
            "noted that",
            "i've noted",
            "i have noted"
        ]

        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()

        # Check if any of the phrases are in the message
        for phrase in memory_phrases:
            if phrase in message_lower:
                # Print statement for visibility
                print(f"\n[MEMORY INTEGRATION] Memory storage intent detected: '{phrase}' in message: '{message[:100]}...'")
                logger.info(f"Memory storage intent detected: '{phrase}'")
                return True

        return False

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
            "let me think",
            "let me see",
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
            "let me see if i can find",
            "let me check",
            "i remember",
            "you mentioned",
            "you told me",
            "you said",
            "as i recall",
            "if i recall",
            "from our previous",
            "from our conversation",
            "from what you told me",
            "based on what you told me",
            "according to what you said",
            "let me see what i know",
            "let me check what i know",
            "let me recall what you said",
            "let me see what you told me",
            "where you live",
            "where you are",
            "your location",
            "where you're from",
            "where you are from",
            "where you reside",
            "where you're located"
        ]

        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()

        # Check for common location-related questions that might be asked in voice mode
        # These are more flexible patterns to match voice transcripts
        location_patterns = [
            "know where i live",
            "know where i am",
            "know my location",
            "know my address",
            "know where i reside",
            "know where i'm from",
            "know where i'm located",
            "remember where i live",
            "remember my location",
            "remember where i'm from",
            "remember my address",
            "you know where i live",
            "you know my location",
            "you know my address",
            "you know where i'm from",
            "you know where i reside",
            "you know where i'm located",
            "do you know where i live",
            "do you know my location",
            "do you know my address",
            "do you know where i'm from",
            "do you remember where i live",
            "do you remember my location",
            "do you remember my address",
            "where do i live",
            "where am i from",
            "where am i located",
            "where is my home",
            "what's my location",
            "what is my location",
            "what's my address",
            "what is my address",
            "what city do i live in",
            "what city am i from",
            "what city am i in",
            "what state do i live in",
            "what state am i from",
            "what state am i in",
            "what country do i live in",
            "what country am i from",
            "what country am i in"
        ]

        # Check for location patterns first (more specific)
        for pattern in location_patterns:
            if pattern in message_lower:
                print(f"\n[MEMORY RETRIEVAL] Location question detected: '{pattern}' in '{message[:50]}...'")
                logger.info(f"Location question detected: '{pattern}'")
                return True

        # Check if the message contains 'where' and 'live', 'reside', etc.
        if ("where" in message_lower and
            ("live" in message_lower or "from" in message_lower or "located" in message_lower or
             "location" in message_lower or "address" in message_lower or "reside" in message_lower)):
            print(f"\n[MEMORY RETRIEVAL] Location question detected in: '{message[:50]}...'")
            logger.info(f"Location question detected in message")
            return True

        # Check if the message contains 'know where' pattern (common in voice transcripts)
        if ("know where" in message_lower or "remember where" in message_lower):
            print(f"\n[MEMORY RETRIEVAL] Knowledge question detected in: '{message[:50]}...'")
            logger.info(f"Knowledge question detected in message")
            return True

        # Check for general retrieval phrases
        for phrase in retrieval_phrases:
            if phrase in message_lower:
                print(f"\n[MEMORY RETRIEVAL] Memory retrieval intent detected: '{phrase}' in '{message[:50]}...'")
                logger.info(f"Memory retrieval intent detected: '{phrase}'")
                return True

        print(f"\n[MEMORY RETRIEVAL] No memory retrieval intent detected in: '{message[:50]}...'")
        return False

    async def store_memory_directly(self, content: str, importance: int = 8, category: str = "preference") -> str:
        """Store a memory directly without going through the orchestrator.

        Args:
            content: The content of the memory
            importance: Importance score from 1-10
            category: Category of the memory

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Create a session
            session = self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user")

            # Set session state
            session.state["current_role"] = "assistant"
            session.state["conversation_mode"] = "direct"

            # Call the memory agent's _handle_store_memory method directly
            result = await self.memory_agent._handle_store_memory(
                content=content,
                importance=importance,
                category=category,
                tool_context=session
            )

            # Print the result
            print(f"\n[MEMORY INTEGRATION] Directly stored memory:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}")

            return result.get("message", "Unknown result")
        except Exception as e:
            logger.error(f"Error storing memory directly: {e}")
            print(f"\n[MEMORY INTEGRATION] Error storing memory directly: {e}")
            return f"Error: {str(e)}"

    async def deduplicate_memories(self, similarity_threshold: float = 0.85):
        """Scan the database for similar memories and remove duplicates.

        Args:
            similarity_threshold: Threshold for considering memories as duplicates (0.0 to 1.0)

        Returns:
            Number of duplicates removed
        """
        if not self.memory_agent or not self.memory_agent.db_connector or not self.memory_agent.db_connector.connected:
            logger.error("Cannot deduplicate memories: Database not connected")
            return 0

        try:
            # Import the memory manager to use list_memories
            from memory.memory_utils import MemoryManager
            manager = MemoryManager()
            if not manager.connect():
                logger.error("Failed to connect to Supabase for memory deduplication")
                return 0

            # Get all memories for the current user
            memories = manager.list_memories(limit=100, user_id=self.user_id)
            if not memories:
                logger.info(f"No memories found for user {self.user_id}")
                return 0

            logger.info(f"Scanning {len(memories)} memories for duplicates")
            print(f"\n[MEMORY DEDUPLICATION] Scanning {len(memories)} memories for duplicates...")

            # Track memories to delete
            duplicates_to_remove = []

            # Compare each memory with all others
            for i, memory1 in enumerate(memories):
                # Skip if this memory is already marked for deletion
                if memory1.get('id') in duplicates_to_remove:
                    continue

                for memory2 in memories[i+1:]:
                    # Skip if this memory is already marked for deletion
                    if memory2.get('id') in duplicates_to_remove:
                        continue

                    # Get importance values for later comparison if needed
                    importance1 = memory1.get('importance', 0)
                    importance2 = memory2.get('importance', 0)

                    # Check if memories are similar
                    content1 = memory1.get('content', '')
                    content2 = memory2.get('content', '')
                    embedding1 = memory1.get('embedding', [])
                    embedding2 = memory2.get('embedding', [])

                    # No longer skipping location-related memories during deduplication

                    # Use the _is_similar_content method from memory_agent
                    is_similar = self.memory_agent._is_similar_content(
                        content1=content1,
                        content2=content2,
                        embedding1=embedding1,
                        embedding2=embedding2,
                        threshold=similarity_threshold
                    )

                    if is_similar:
                        # Determine which memory to keep (higher importance or more recent)
                        importance1 = memory1.get('importance', 0)
                        importance2 = memory2.get('importance', 0)
                        timestamp1 = memory1.get('timestamp', '')
                        timestamp2 = memory2.get('timestamp', '')

                        # Keep the memory with higher importance, or if equal, the more recent one
                        if importance1 > importance2:
                            memory_to_remove = memory2.get('id')
                            memory_to_keep = memory1.get('id')
                        elif importance2 > importance1:
                            memory_to_remove = memory1.get('id')
                            memory_to_keep = memory2.get('id')
                        else:
                            # Equal importance, keep the more recent one
                            if timestamp1 > timestamp2:
                                memory_to_remove = memory2.get('id')
                                memory_to_keep = memory1.get('id')
                            else:
                                memory_to_remove = memory1.get('id')
                                memory_to_keep = memory2.get('id')

                        # Add to list of duplicates to remove
                        if memory_to_remove not in duplicates_to_remove:
                            duplicates_to_remove.append(memory_to_remove)
                            print(f"  Found duplicate: '{content1[:50]}...' and '{content2[:50]}...'")
                            print(f"  Keeping memory {memory_to_keep}, removing {memory_to_remove}")

            # Remove duplicates
            if duplicates_to_remove:
                # Create a backup of memories to be deleted
                deleted_memories = []
                for memory_id in duplicates_to_remove:
                    # Find the memory in the original list
                    memory_to_delete = next((m for m in memories if m.get('id') == memory_id), None)
                    if memory_to_delete:
                        deleted_memories.append(memory_to_delete)
                        print(f"\n[MEMORY DEDUPLICATION] Deleting memory: ID={memory_id}, Content='{memory_to_delete.get('content', '')}', Importance={memory_to_delete.get('importance', 0)}")

                    # Delete the memory
                    self.memory_agent.db_connector.delete_memory(memory_id)

                # Log detailed information about deleted memories
                for i, memory in enumerate(deleted_memories):
                    logger.info(f"Deleted memory {i+1}: ID={memory.get('id')}, Content='{memory.get('content', '')}', Importance={memory.get('importance', 0)}")

                logger.info(f"Removed {len(duplicates_to_remove)} duplicate memories")
                print(f"\n[MEMORY DEDUPLICATION] Removed {len(duplicates_to_remove)} duplicate memories")
                return len(duplicates_to_remove)
            else:
                logger.info("No duplicate memories found")
                print("\n[MEMORY DEDUPLICATION] No duplicate memories found")
                return 0

        except Exception as e:
            logger.error(f"Error deduplicating memories: {e}")
            print(f"\n[MEMORY DEDUPLICATION] Error: {e}")
            return 0

    async def cleanup(self):
        """Clean up resources used by the memory integration."""
        # Clean up the memory agent
        await self.memory_agent.cleanup()
        logger.info("Memory integration cleanup complete")
