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

from memory_agent import ConversationMemoryAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryIntegration:
    """Integration class for the memory agent."""

    def __init__(self, openai_api_key: str, gemini_api_key: str = None):
        """Initialize the memory integration.

        Args:
            openai_api_key: OpenAI API key for the LLM models
            gemini_api_key: Optional Gemini API key for alternative models
        """
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key

        # Initialize session service
        self.session_service = InMemorySessionService()

        # Initialize models
        self.storage_model = LiteLlm(
            model_name="gpt-4o",
            api_key=openai_api_key,
            temperature=0.2,  # Lower temperature for more consistent memory storage
            max_tokens=1024   # Limit token usage for efficiency
        )

        self.retrieval_model = LiteLlm(
            model_name="gpt-4o",
            api_key=openai_api_key,
            temperature=0.3,  # Slightly higher temperature for retrieval
            max_tokens=1024   # Limit token usage for efficiency
        )

        self.orchestrator_model = LiteLlm(
            model_name="gpt-4o",
            api_key=openai_api_key,
            temperature=0.1,  # Very low temperature for consistent orchestration decisions
            max_tokens=2048   # Allow more tokens for orchestration
        )

        # Initialize the memory agent
        self.memory_agent = self._create_memory_agent()

        logger.info("Memory integration initialized")

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
            session_service=self.session_service
        )

        return memory_agent

    async def process_user_message(self, message: str, conversation_mode: str) -> None:
        """Process a user message and store it in memory if relevant.

        Args:
            message: The user's message
            conversation_mode: 'audio' or 'chat'
        """
        await self.memory_agent.process_message(message, "user", conversation_mode)

    async def process_assistant_message(self, message: str, conversation_mode: str) -> None:
        """Process an assistant message and store it in memory if relevant.

        Args:
            message: The assistant's message
            conversation_mode: 'audio' or 'chat'
        """
        await self.memory_agent.process_message(message, "assistant", conversation_mode)

    async def get_context_with_memories(self, current_message: str) -> str:
        """Get relevant memories for the current message and format them for context.

        Args:
            current_message: The current message to find relevant memories for

        Returns:
            Formatted string of memories for inclusion in the prompt
        """
        try:
            # Create a dedicated session for this operation
            session = self.session_service.create_session()

            # Store the current message in the session for context
            session.set("current_message", current_message)

            # Get relevant memories using the memory agent
            relevant_memories = await self.memory_agent.get_relevant_memories(current_message)

            # Format memories for context
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories)

            logger.info(f"Retrieved {len(relevant_memories)} memories for context")
            return memory_context
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
            return ""  # Return empty string on error
        finally:
            # Clean up the session
            if 'session' in locals():
                await session.aclose()

    async def enhance_prompt_with_memories(self, prompt: str, current_message: str) -> str:
        """Enhance a prompt with relevant memories.

        Args:
            prompt: The original prompt
            current_message: The current message to find relevant memories for

        Returns:
            Enhanced prompt with memories
        """
        try:
            # Create a dedicated session for this operation
            session = self.session_service.create_session()

            # Store the current message in the session for context
            session.set("current_message", current_message)

            # Get relevant memories
            relevant_memories = await self.memory_agent.get_relevant_memories(current_message)

            # Format memories for context
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories)

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
            if 'session' in locals():
                await session.aclose()

    async def enhance_chat_messages(self, messages: List[Dict[str, str]], current_message: str) -> List[Dict[str, str]]:
        """Enhance chat messages with relevant memories.

        Args:
            messages: List of chat messages in the format [{"role": "...", "content": "..."}]
            current_message: The current message to find relevant memories for

        Returns:
            Enhanced messages with memories
        """
        try:
            # Create a dedicated session for this operation
            session = self.session_service.create_session()

            # Store the current message in the session for context
            session.set("current_message", current_message)

            # Get relevant memories
            relevant_memories = await self.memory_agent.get_relevant_memories(current_message)

            # Format memories for context
            memory_context = self.memory_agent.format_memories_for_context(relevant_memories)

            if not memory_context:
                return messages  # Return original messages if no memories

            # Create a new system message with memories or enhance existing one
            system_message_exists = False
            enhanced_messages = []

            for message in messages:
                if message["role"] == "system":
                    # Enhance existing system message
                    message["content"] = f"{message['content']}\n\n{memory_context}"
                    system_message_exists = True
                enhanced_messages.append(message)

            # Add a new system message with memories if none exists
            if not system_message_exists:
                enhanced_messages.insert(0, {
                    "role": "system",
                    "content": f"You are a helpful assistant with access to the user's previous conversations. {memory_context}"
                })

            logger.info("Enhanced chat messages with memory context")
            return enhanced_messages
        except Exception as e:
            logger.error(f"Error enhancing chat messages with memories: {e}")
            return messages  # Return original messages on error
        finally:
            # Clean up the session
            if 'session' in locals():
                await session.aclose()

    async def cleanup(self):
        """Clean up resources used by the memory integration."""
        # Clean up the memory agent
        await self.memory_agent.cleanup()
        logger.info("Memory integration cleanup complete")
