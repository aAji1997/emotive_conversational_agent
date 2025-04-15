import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryDatabaseConnector:
    """Connector for storing and retrieving memories from a vector database."""

    def __init__(self, supabase_url=None, supabase_key=None):
        """Initialize the memory database connector.

        Args:
            supabase_url: URL for the Supabase instance
            supabase_key: API key for the Supabase instance
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.connected = False
        self.memories = []  # Temporary in-memory storage until Supabase is set up

        # Try to connect to Supabase if credentials are provided
        if supabase_url and supabase_key:
            try:
                # Import supabase here to avoid dependency issues if not installed
                from supabase import create_client
                self.client = create_client(supabase_url, supabase_key)
                self.connected = True
                logger.info("Connected to Supabase vector database")
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                logger.info("Using in-memory storage for memories instead")
        else:
            logger.info("No Supabase credentials provided, using in-memory storage")

    def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store a memory in the database.

        Args:
            memory_data: Dictionary containing memory information
                - content: The actual memory content
                - embedding: Vector embedding of the memory
                - metadata: Additional metadata about the memory
                - timestamp: When the memory was created
                - source: Source of the memory (user, assistant, etc.)

        Returns:
            memory_id: ID of the stored memory
        """
        memory_id = str(time.time())  # Simple timestamp-based ID
        memory_data['id'] = memory_id

        if self.connected:
            try:
                # Store in Supabase
                self.client.table('memories').insert(memory_data).execute()
                logger.info(f"Stored memory in Supabase with ID: {memory_id}")
            except Exception as e:
                logger.error(f"Failed to store memory in Supabase: {e}")
                # Fall back to in-memory storage
                self.memories.append(memory_data)
                logger.info(f"Stored memory in-memory with ID: {memory_id}")
        else:
            # Store in-memory
            self.memories.append(memory_data)
            logger.info(f"Stored memory in-memory with ID: {memory_id}")

        return memory_id

    def retrieve_memories(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories similar to the query embedding.

        Args:
            query_embedding: Vector embedding to search for
            limit: Maximum number of memories to return

        Returns:
            List of memory dictionaries
        """
        if self.connected:
            try:
                # Search in Supabase using vector similarity
                result = self.client.rpc(
                    'match_memories',
                    {'query_embedding': query_embedding, 'match_threshold': 0.7, 'match_count': limit}
                ).execute()

                return result.data
            except Exception as e:
                logger.error(f"Failed to retrieve memories from Supabase: {e}")
                # Fall back to in-memory search
                logger.info("Falling back to in-memory search")

        # Simple in-memory search (no actual vector similarity)
        # In a real implementation, you would compute cosine similarity between embeddings
        # For now, just return the most recent memories
        sorted_memories = sorted(self.memories, key=lambda x: x.get('timestamp', 0), reverse=True)
        return sorted_memories[:limit]

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: ID of the memory to retrieve

        Returns:
            Memory dictionary or None if not found
        """
        if self.connected:
            try:
                result = self.client.table('memories').select('*').eq('id', memory_id).execute()
                if result.data:
                    return result.data[0]
                return None
            except Exception as e:
                logger.error(f"Failed to retrieve memory from Supabase: {e}")
                # Fall back to in-memory search

        # Search in-memory
        for memory in self.memories:
            if memory.get('id') == memory_id:
                return memory
        return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from the database.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            True if successful, False otherwise
        """
        if self.connected:
            try:
                self.client.table('memories').delete().eq('id', memory_id).execute()
                logger.info(f"Deleted memory from Supabase with ID: {memory_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete memory from Supabase: {e}")
                # Fall back to in-memory deletion

        # Delete from in-memory storage
        for i, memory in enumerate(self.memories):
            if memory.get('id') == memory_id:
                del self.memories[i]
                logger.info(f"Deleted memory from in-memory storage with ID: {memory_id}")
                return True

        logger.warning(f"Memory with ID {memory_id} not found")
        return False

class ConversationMemoryAgent:
    def __init__(self, storage_agent_model, retrieval_agent_model, orchestrator_agent_model, session_service):
        """
        This agent will be responsible for determining which information about a user or events that occurred during a real-time or text conversation should be saved as a memory embedding for future reference.
        The agent will also be responsible for retrieving relevant memories when needed. Memories will be stored in a supabase vector database via the pgvector extension.
        This agent will in-turn be composed of three sub-agents: one for storing memories, one for retrieving memories, and an orchestrator agent that will determine when
          to store and retrieve memories, as well as whether stored or retrieved memories are sufficiently relevant to the current conversation.
        """
        self.storage_agent_model = storage_agent_model
        self.retrieval_agent_model = retrieval_agent_model
        self.orchestrator_agent_model = orchestrator_agent_model
        self.session_service = session_service

        # Initialize the database connector
        self.db_connector = MemoryDatabaseConnector()

        # Initialize conversation history
        self.conversation_history = []

        # Define tools for the agents
        self.store_memory_tool = {
            "name": "store_memory",
            "description": "Store a new memory in the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the memory"
                    },
                    "importance": {
                        "type": "integer",
                        "description": "Importance score from 1-10"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the memory (preference, personal_info, event, etc.)"
                    }
                },
                "required": ["content", "importance", "category"]
            }
        }

        self.retrieve_memories_tool = {
            "name": "retrieve_memories",
            "description": "Retrieve memories relevant to a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in memories"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of memories to retrieve"
                    }
                },
                "required": ["query"]
            }
        }

        # Set up the agents
        self.storage_agent = Agent(
            name="memory_storage",
            model=storage_agent_model,
            instruction="""You are a memory storage agent. Your job is to determine what information from a conversation should be stored as a memory.
            Focus on important facts about the user, their preferences, and key events. Create concise, well-formatted memories that will be useful in future conversations.

            When storing a memory:
            1. Extract key information from the conversation
            2. Format it as a concise, factual statement
            3. Assign an importance score (1-10) based on how useful this information will be in future conversations
            4. Categorize the memory appropriately (preference, personal_info, event, etc.)
            5. Only store information that was explicitly stated or can be directly inferred

            Do not store general conversation elements, only specific, useful information about the user.""",
            description="Agent responsible for determining what information to store as memories",
            tools=[self.store_memory_tool]
        )

        self.retrieval_agent = Agent(
            name="memory_retrieval",
            model=retrieval_agent_model,
            instruction="""You are a memory retrieval agent. Your job is to find relevant memories based on the current conversation context.
            Determine which memories would be most helpful for the assistant to provide a personalized and contextually appropriate response.

            When retrieving memories:
            1. Analyze the current conversation to identify key topics, entities, and user needs
            2. Formulate specific queries to search for relevant memories
            3. Evaluate the relevance of retrieved memories to the current context
            4. Only return memories that would be helpful for the current conversation

            Focus on quality over quantity - it's better to return a few highly relevant memories than many loosely related ones.""",
            description="Agent responsible for retrieving relevant memories based on conversation context",
            tools=[self.retrieve_memories_tool]
        )

        # Define the invoke_agent tool for the orchestrator
        self.invoke_agent_tool = {
            "name": "invoke_agent",
            "description": "Invoke a sub-agent to perform a specific task",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent to invoke",
                        "enum": ["memory_storage", "memory_retrieval"]
                    },
                    "input": {
                        "type": "string",
                        "description": "Input to provide to the agent"
                    }
                },
                "required": ["agent_name", "input"]
            }
        }

        self.orchestrator_agent = Agent(
            name="memory_orchestrator",
            model=orchestrator_agent_model,
            instruction="""You are a memory orchestrator. You coordinate the storage and retrieval of memories during conversations.

            Your responsibilities include:
            1. Determining when to store new memories by invoking the memory_storage agent
            2. Determining when to retrieve relevant memories by invoking the memory_retrieval agent
            3. Evaluating whether stored or retrieved memories are sufficiently relevant to the current conversation

            Guidelines:
            - Only store important information that would be useful in future conversations
            - Only retrieve memories that are directly relevant to the current conversation topic
            - Consider the context and flow of the conversation when making decisions
            - Balance between storing too much (information overload) and too little (missing important details)

            You have access to two sub-agents:
            1. memory_storage: Determines what information to store as memories
            2. memory_retrieval: Finds relevant memories based on conversation context

            Use the invoke_agent tool to delegate tasks to these specialized agents.""",
            description="Main Orchestrator Agent for Memory Management",
            sub_agents=[self.storage_agent, self.retrieval_agent],
            tools=[self.invoke_agent_tool]
        )

    async def process_message(self, message: str, role: str, conversation_mode: str) -> None:
        """Process a new message from the conversation.

        Args:
            message: The message content
            role: 'user' or 'assistant'
            conversation_mode: 'audio' or 'chat'
        """
        # Add to conversation history
        timestamp = datetime.now().isoformat()
        message_entry = {
            "role": role,
            "content": message,
            "timestamp": timestamp,
            "conversation_mode": conversation_mode
        }
        self.conversation_history.append(message_entry)

        # Limit history size to prevent context overflow
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

        # Check if we should store this as a memory
        await self._check_and_store_memory(message, role, conversation_mode)

    async def get_relevant_memories(self, current_message: str) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current message.

        Args:
            current_message: The current message to find relevant memories for

        Returns:
            List of relevant memory dictionaries
        """
        # Create a context from recent conversation history
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Create a session with context
        retrieval_session = self.session_service.create_session()

        # Register tool handlers
        retrieval_session.register_tool_handler("retrieve_memories", self._handle_retrieve_memories)

        # Create the retrieval runner
        retrieval_runner = Runner(agent=self.retrieval_agent, session=retrieval_session)

        # Create the prompt for the retrieval agent
        retrieval_prompt = f"""Based on the following conversation, retrieve relevant memories that would be helpful for responding to the current message.

        Current conversation:
        {context}

        Current message: {current_message}

        Use the retrieve_memories tool to search for relevant memories. Formulate a specific, targeted query that will find the most relevant information."""

        try:
            # Run the retrieval agent
            retrieval_response = await retrieval_runner.run_async(retrieval_prompt)
            logger.info(f"Retrieval agent response: {retrieval_response.text[:100]}...")

            # The retrieval agent should have used the retrieve_memories tool,
            # which would have stored the results in the database connector
            # We'll retrieve the most recent memories as a fallback
            dummy_embedding = [0.1] * 1536  # Placeholder embedding
            relevant_memories = self.db_connector.retrieve_memories(dummy_embedding, limit=3)

            return relevant_memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []  # Return empty list on error
        finally:
            # Clean up the session to avoid memory leaks
            await retrieval_session.aclose()

    # Tool handlers
    async def _handle_store_memory(self, params, session):
        """Handle the store_memory tool call.

        Args:
            params: Parameters from the tool call
            session: The current session

        Returns:
            Response message
        """
        try:
            content = params.get("content")
            importance = params.get("importance")
            category = params.get("category")

            if not content or not importance or not category:
                return "Error: Missing required parameters. Please provide content, importance, and category."

            # Create memory data
            memory_data = {
                "content": content,
                "importance": importance,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "source": session.get("current_role", "unknown"),
                "conversation_mode": session.get("conversation_mode", "unknown"),
                "embedding": [0.1] * 1536  # Placeholder embedding
            }

            # Store the memory
            memory_id = self.db_connector.store_memory(memory_data)
            logger.info(f"Stored new memory with ID: {memory_id}")

            return f"Successfully stored memory: {content}"
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"Error storing memory: {str(e)}"

    async def _handle_retrieve_memories(self, params, session):  # session parameter is required by ADK but not used here
        """Handle the retrieve_memories tool call.

        Args:
            params: Parameters from the tool call
            session: The current session

        Returns:
            Response message with retrieved memories
        """
        try:
            query = params.get("query")
            limit = params.get("limit", 3)  # Default to 3 memories

            if not query:
                return "Error: Missing required parameter 'query'."

            # In a real implementation, you would generate an embedding for the query
            # and use it to search the vector database
            query_embedding = [0.1] * 1536  # Placeholder embedding

            # Retrieve memories
            memories = self.db_connector.retrieve_memories(query_embedding, limit=limit)

            # Format memories for response
            if not memories:
                return "No relevant memories found."

            formatted_memories = ["Retrieved memories:"]
            for i, memory in enumerate(memories):
                content = memory.get('content', 'No content available')
                category = memory.get('category', 'general')
                importance = memory.get('importance', 0)

                formatted_memories.append(f"{i+1}. [{category}] {content} (Importance: {importance})")

            return "\n".join(formatted_memories)
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return f"Error retrieving memories: {str(e)}"

    async def _check_and_store_memory(self, message: str, role: str, conversation_mode: str) -> None:
        """Check if a message should be stored as a memory and store it if needed.

        Args:
            message: The message content
            role: 'user' or 'assistant'
            conversation_mode: 'audio' or 'chat'
        """
        # Create a context from recent conversation history
        recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Create a session with context
        orchestrator_session = self.session_service.create_session()
        # Store the current role and conversation mode in the session
        orchestrator_session.set("current_role", role)
        orchestrator_session.set("conversation_mode", conversation_mode)

        # Register tool handlers
        orchestrator_session.register_tool_handler("store_memory", self._handle_store_memory)
        orchestrator_session.register_tool_handler("retrieve_memories", self._handle_retrieve_memories)

        # Create the orchestrator runner
        orchestrator_runner = Runner(agent=self.orchestrator_agent, session=orchestrator_session)

        # Create the prompt for the orchestrator
        orchestrator_prompt = f"""Analyze the following conversation and determine if any important information should be stored as a memory.

        Conversation context:
        {context}

        Message to evaluate:
        {role}: {message}

        If you find important information about the user, their preferences, or key events that would be useful in future conversations, invoke the memory_storage agent to create a memory.
        Otherwise, explain why no memory needs to be stored."""

        try:
            # Run the orchestrator
            orchestrator_response = await orchestrator_runner.run_async(orchestrator_prompt)
            logger.info(f"Orchestrator response: {orchestrator_response.text[:100]}...")
        except Exception as e:
            logger.error(f"Error running orchestrator agent: {e}")
        finally:
            # Clean up the session to avoid memory leaks
            await orchestrator_session.aclose()

    def format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for inclusion in the conversation context.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""

        formatted_memories = ["### Relevant memories from previous conversations:"]

        for i, memory in enumerate(memories):
            content = memory.get('content', 'No content available')
            timestamp = memory.get('timestamp', 'Unknown time')
            category = memory.get('category', 'general')

            # Convert ISO timestamp to readable format if possible
            try:
                dt = datetime.fromisoformat(timestamp)
                readable_time = dt.strftime("%B %d, %Y")
            except (ValueError, TypeError):
                readable_time = timestamp

            formatted_memories.append(f"{i+1}. [{category}] {content} (From {readable_time})")

        formatted_memories.append("###\n")
        return "\n".join(formatted_memories)

    async def cleanup(self):
        """Clean up resources used by the memory agent."""
        # Currently, there's not much to clean up since sessions are cleaned up after each use
        # This method is provided for future extensions
        logger.info("Memory agent cleanup complete")
