"""
Traced Memory Agent Implementation.

This module provides a traced version of the ConversationMemoryAgent that
adds Weave tracing to visualize the interactions between memory components.
"""

import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Import the original memory agent
from memory.memory_agent import ConversationMemoryAgent, MemoryDatabaseConnector

# Import tracing functionality
from memory.memory_tracing import (
    trace_method,
    trace_interaction,
    initialize_tracing,
    end_tracing,
    create_trace_visualization
)

# Set up logging
logger = logging.getLogger(__name__)

class TracedMemoryDatabaseConnector(MemoryDatabaseConnector):
    """A traced version of the MemoryDatabaseConnector."""

    @trace_method("database", "create_embedding")
    def create_embedding(self, text: str) -> List[float]:
        """Traced version of create_embedding."""
        return super().create_embedding(text)

    @trace_method("database", "store")
    def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Traced version of store_memory."""
        return super().store_memory(memory_data)

    @trace_method("database", "retrieve")
    def retrieve_memories(self, query_embedding: List[float], limit: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """Traced version of retrieve_memories."""
        return super().retrieve_memories(query_embedding, limit, user_id)

class TracedConversationMemoryAgent(ConversationMemoryAgent):
    """
    A traced version of the ConversationMemoryAgent that adds Weave tracing
    to visualize the interactions between memory components.
    """

    def __init__(self, storage_agent_model, retrieval_agent_model, orchestrator_agent_model,
                 session_service, user_id=None, supabase_url=None, supabase_key=None,
                 enable_tracing=True, trace_project_name="memory-agent-tracing"):
        """
        Initialize the traced conversation memory agent.

        Args:
            storage_agent_model: Model for the storage agent
            retrieval_agent_model: Model for the retrieval agent
            orchestrator_agent_model: Model for the orchestrator agent
            session_service: Session service for the agents
            user_id: Optional user ID
            supabase_url: Supabase URL for the database
            supabase_key: Supabase API key
            enable_tracing: Whether to enable tracing
            trace_project_name: Project name for the traces
        """
        # Initialize tracing
        self.tracing_enabled = enable_tracing
        if enable_tracing:
            run_name = f"memory-trace-{user_id or 'anonymous'}-{int(time.time())}"
            initialize_tracing(project_name=trace_project_name, run_name=run_name, enabled=True)

        # Initialize the database connector with tracing
        self.db_connector = TracedMemoryDatabaseConnector(supabase_url, supabase_key)

        # Call the parent constructor with our traced database connector
        super().__init__(
            storage_agent_model=storage_agent_model,
            retrieval_agent_model=retrieval_agent_model,
            orchestrator_agent_model=orchestrator_agent_model,
            session_service=session_service,
            user_id=user_id,
            supabase_url=supabase_url,
            supabase_key=supabase_key
        )

        # Replace the database connector with our traced version
        self.db_connector = TracedMemoryDatabaseConnector(supabase_url, supabase_key)

        logger.info("Traced memory agent initialized")

    @trace_method("orchestrator", "process_message")
    async def process_message(self, message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Traced version of process_message that adds tracing for component interactions.

        Args:
            message: The message to process

        Returns:
            Tuple of (response, retrieved_memories)
        """
        # Trace the start of processing
        trace_interaction("user", "orchestrator", "message", {"message": message})

        # Call the orchestrator to decide what to do
        trace_interaction("orchestrator", "orchestrator_agent", "request", {"message": message})
        orchestrator_decision = await self._call_orchestrator_agent(message)
        trace_interaction("orchestrator_agent", "orchestrator", "response", {"decision": orchestrator_decision})

        retrieved_memories = []

        # If the orchestrator decides to retrieve memories
        if orchestrator_decision.get("should_retrieve", False):
            # Trace the retrieval request
            trace_interaction("orchestrator", "retrieval", "request", {"message": message})

            # Call the retrieval agent
            retrieval_result = await self._call_retrieval_agent(message)

            # Trace the retrieval response
            trace_interaction("retrieval", "orchestrator", "response", {"result": retrieval_result})

            # Get the memories from the database
            if retrieval_result.get("should_retrieve", False):
                query = retrieval_result.get("query", message)
                limit = retrieval_result.get("limit", 5)

                # Trace the database retrieval
                trace_interaction("retrieval", "database", "request", {"query": query, "limit": limit})

                # Create embedding for the query
                query_embedding = self.db_connector.create_embedding(query)

                # Retrieve memories from the database
                retrieved_memories = self.db_connector.retrieve_memories(
                    query_embedding=query_embedding,
                    limit=limit,
                    user_id=self.user_id
                )

                # Trace the database response
                trace_interaction("database", "retrieval", "response", {"memories_count": len(retrieved_memories)})

        # If the orchestrator decides to store a memory
        if orchestrator_decision.get("should_store", False):
            # Trace the storage request
            trace_interaction("orchestrator", "storage", "request", {"message": message})

            # Call the storage agent
            storage_result = await self._call_storage_agent(message)

            # Trace the storage response
            trace_interaction("storage", "orchestrator", "response", {"result": storage_result})

            # Store the memory in the database
            if storage_result.get("should_store", False):
                content = storage_result.get("content", "")
                importance = storage_result.get("importance", 5)
                category = storage_result.get("category", "general")

                # Trace the database storage
                trace_interaction("storage", "database", "request", {
                    "content": content,
                    "importance": importance,
                    "category": category
                })

                # Create memory data dictionary
                memory_data = {
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                    "source": "storage_agent",
                    "conversation_mode": "test",
                    "user_id": self.user_id
                }

                # Store the memory
                memory_id = self.db_connector.store_memory(memory_data)

                # Trace the database response
                trace_interaction("database", "storage", "response", {"memory_id": memory_id})

        # Generate a response based on the message and retrieved memories
        response = f"Processed message: {message}"
        if retrieved_memories:
            response += f"\nRetrieved {len(retrieved_memories)} memories."

        # Trace the response back to the user
        trace_interaction("orchestrator", "user", "response", {"response": response})

        return response, retrieved_memories

    @trace_method("orchestrator", "call_orchestrator")
    async def _call_orchestrator_agent(self, message: str) -> Dict[str, Any]:
        """Traced version of orchestrator agent call.

        This method creates a session and runs the orchestrator agent to decide
        whether to store or retrieve memories based on the message.

        Args:
            message: The message to process

        Returns:
            Dictionary with orchestrator decision
        """
        # Create a session for the orchestrator agent
        import uuid
        from google.genai import types

        orchestrator_session_id = str(uuid.uuid4())
        self.session_service.create_session(
            app_name="memory_agent",
            user_id=self.user_id or "test_user",
            session_id=orchestrator_session_id
        )

        # Create a runner for the orchestrator agent
        from google.adk.runners import Runner
        orchestrator_runner = Runner(
            agent=self.orchestrator_agent,
            app_name="memory_agent",
            session_service=self.session_service
        )

        # Create the prompt for the orchestrator
        orchestrator_prompt = f"""Analyze this message and decide whether to store it as a memory, retrieve relevant memories, or both.

        Message: {message}

        Respond with a JSON object containing:
        - should_store: true/false - whether to store this as a memory
        - should_retrieve: true/false - whether to retrieve relevant memories
        """

        # Run the orchestrator agent
        content = types.Content(role='user', parts=[types.Part(text=orchestrator_prompt)])
        events = orchestrator_runner.run(
            user_id=self.user_id or "test_user",
            session_id=orchestrator_session_id,
            new_message=content
        )

        # Process the response
        orchestrator_response = None
        for event in events:
            if event.is_final_response():
                orchestrator_response = event.content.parts[0].text
                break

        # Parse the response to get the decision
        import json
        import re

        # Try to extract JSON from the response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{[\s\S]*\}', orchestrator_response)
            if json_match:
                json_str = json_match.group(0)
                decision = json.loads(json_str)
            else:
                # Fallback to a default decision
                decision = {
                    "should_store": "store" in orchestrator_response.lower(),
                    "should_retrieve": "retrieve" in orchestrator_response.lower()
                }
        except Exception as e:
            # If JSON parsing fails, make a simple decision based on keywords
            decision = {
                "should_store": "store" in orchestrator_response.lower(),
                "should_retrieve": "retrieve" in orchestrator_response.lower()
            }

        return decision

    @trace_method("storage", "call_storage")
    async def _call_storage_agent(self, message: str) -> Dict[str, Any]:
        """Traced version of storage agent call.

        This method creates a session and runs the storage agent to determine
        what information from the message should be stored as a memory.

        Args:
            message: The message to process

        Returns:
            Dictionary with storage decision
        """
        # Create a session for the storage agent
        import uuid
        from google.genai import types

        storage_session_id = str(uuid.uuid4())
        self.session_service.create_session(
            app_name="memory_agent",
            user_id=self.user_id or "test_user",
            session_id=storage_session_id
        )

        # Create a runner for the storage agent
        from google.adk.runners import Runner
        storage_runner = Runner(
            agent=self.storage_agent,
            app_name="memory_agent",
            session_service=self.session_service
        )

        # Create the prompt for the storage agent
        storage_prompt = f"""Analyze this message and determine if it contains information that should be stored as a memory.

        Message: {message}

        If the message contains important information about the user, their preferences, or key events,
        extract that information and format it as a concise, factual statement.

        Respond with a JSON object containing:
        - should_store: true/false - whether to store this as a memory
        - content: the formatted memory content (if should_store is true)
        - importance: a score from 1-10 indicating the importance of this memory (if should_store is true)
        - category: the category of the memory (preference, personal_info, event, etc.) (if should_store is true)
        """

        # Run the storage agent
        content = types.Content(role='user', parts=[types.Part(text=storage_prompt)])
        events = storage_runner.run(
            user_id=self.user_id or "test_user",
            session_id=storage_session_id,
            new_message=content
        )

        # Process the response
        storage_response = None
        for event in events:
            if event.is_final_response():
                storage_response = event.content.parts[0].text
                break

        # Parse the response to get the decision
        import json
        import re

        # Try to extract JSON from the response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{[\s\S]*\}', storage_response)
            if json_match:
                json_str = json_match.group(0)
                decision = json.loads(json_str)
            else:
                # Fallback to a default decision
                decision = {
                    "should_store": "store" in storage_response.lower() and "not" not in storage_response.lower()[:20],
                    "content": message,
                    "importance": 5,
                    "category": "general"
                }
        except Exception:
            # If JSON parsing fails, make a simple decision based on keywords
            decision = {
                "should_store": "store" in storage_response.lower() and "not" not in storage_response.lower()[:20],
                "content": message,
                "importance": 5,
                "category": "general"
            }

        return decision

    @trace_method("retrieval", "call_retrieval")
    async def _call_retrieval_agent(self, message: str) -> Dict[str, Any]:
        """Traced version of retrieval agent call.

        This method creates a session and runs the retrieval agent to determine
        what memories should be retrieved based on the message.

        Args:
            message: The message to process

        Returns:
            Dictionary with retrieval decision
        """
        # Create a session for the retrieval agent
        import uuid
        from google.genai import types

        retrieval_session_id = str(uuid.uuid4())
        self.session_service.create_session(
            app_name="memory_agent",
            user_id=self.user_id or "test_user",
            session_id=retrieval_session_id
        )

        # Create a runner for the retrieval agent
        from google.adk.runners import Runner
        retrieval_runner = Runner(
            agent=self.retrieval_agent,
            app_name="memory_agent",
            session_service=self.session_service
        )

        # Create the prompt for the retrieval agent
        retrieval_prompt = f"""Analyze this message and determine if it requires retrieving memories to provide a good response.

        Message: {message}

        If the message is asking about something the user has mentioned before, or if it would be helpful to recall
        previous information about the user to provide a personalized response, we should retrieve relevant memories.

        Respond with a JSON object containing:
        - should_retrieve: true/false - whether to retrieve memories
        - query: a specific query to use for retrieving memories (if should_retrieve is true)
        - limit: the maximum number of memories to retrieve (if should_retrieve is true)
        """

        # Run the retrieval agent
        content = types.Content(role='user', parts=[types.Part(text=retrieval_prompt)])
        events = retrieval_runner.run(
            user_id=self.user_id or "test_user",
            session_id=retrieval_session_id,
            new_message=content
        )

        # Process the response
        retrieval_response = None
        for event in events:
            if event.is_final_response():
                retrieval_response = event.content.parts[0].text
                break

        # Parse the response to get the decision
        import json
        import re

        # Try to extract JSON from the response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\{[\s\S]*\}', retrieval_response)
            if json_match:
                json_str = json_match.group(0)
                decision = json.loads(json_str)
            else:
                # Fallback to a default decision
                decision = {
                    "should_retrieve": "retrieve" in retrieval_response.lower() and "not" not in retrieval_response.lower()[:20],
                    "query": message,
                    "limit": 5
                }
        except Exception:
            # If JSON parsing fails, make a simple decision based on keywords
            decision = {
                "should_retrieve": "retrieve" in retrieval_response.lower() and "not" not in retrieval_response.lower()[:20],
                "query": message,
                "limit": 5
            }

        return decision

    def close(self):
        """Close the agent and end tracing."""
        if self.tracing_enabled:
            end_tracing()

    def generate_visualization(self, output_file="memory_trace_visualization.html"):
        """Generate a visualization of the traced interactions."""
        if self.tracing_enabled:
            return create_trace_visualization(output_file)
        return None
