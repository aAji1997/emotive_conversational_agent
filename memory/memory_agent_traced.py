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
    
    @trace_method("database", "store")
    def store_memory(self, content, importance, category, user_id=None):
        """Traced version of store_memory."""
        return super().store_memory(content, importance, category, user_id)
    
    @trace_method("database", "retrieve")
    def retrieve_memories(self, query, limit=5, user_id=None):
        """Traced version of retrieve_memories."""
        return super().retrieve_memories(query, limit, user_id)

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
                
                # Retrieve memories from the database
                retrieved_memories = self.db_connector.retrieve_memories(
                    query=query,
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
                
                # Store the memory
                memory_id = self.db_connector.store_memory(
                    content=content,
                    importance=importance,
                    category=category,
                    user_id=self.user_id
                )
                
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
        """Traced version of _call_orchestrator_agent."""
        return await super()._call_orchestrator_agent(message)
    
    @trace_method("storage", "call_storage")
    async def _call_storage_agent(self, message: str) -> Dict[str, Any]:
        """Traced version of _call_storage_agent."""
        return await super()._call_storage_agent(message)
    
    @trace_method("retrieval", "call_retrieval")
    async def _call_retrieval_agent(self, message: str) -> Dict[str, Any]:
        """Traced version of _call_retrieval_agent."""
        return await super()._call_retrieval_agent(message)
    
    def close(self):
        """Close the agent and end tracing."""
        if self.tracing_enabled:
            end_tracing()
    
    def generate_visualization(self, output_file="memory_trace_visualization.html"):
        """Generate a visualization of the traced interactions."""
        if self.tracing_enabled:
            return create_trace_visualization(output_file)
        return None
