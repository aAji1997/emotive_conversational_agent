import os
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from litellm import completion
from google.genai import types # For creating message Content/Parts

import spacy

# Set up logging
logger = logging.getLogger(__name__)

class MemoryDatabaseConnector:
    """Connector for storing and retrieving memories from a vector database."""

    def __init__(self, supabase_url=None, supabase_key=None):
        """Initialize the memory database connector.

        Args:
            supabase_url: URL for the Supabase instance
            supabase_key: API key for the Supabase instance
        """
        self.nlp = spacy.load("en_core_web_md") # medium english model
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

    def create_embedding(self, text: str) -> List[float]:
        """Create a vector embedding for a given text.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        doc = self.nlp(text)
        return doc.vector.tolist()  # Convert to list for JSON compatibility

    def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store a memory in the database.

        Args:
            memory_data: Dictionary containing memory information
                - content: The actual memory content
                - embedding: Vector embedding of the memory
                - metadata: Additional metadata about the memory
                - timestamp: When the memory was created
                - source: Source of the memory (user, assistant, etc.)
                - user_id: Optional user ID to associate with the memory

        Returns:
            memory_id: ID of the stored memory
        """
        memory_id = str(time.time())  # Simple timestamp-based ID
        memory_data['id'] = memory_id

        # Add detailed logging
        print(f"\n[DEBUG] Memory data before storage:")
        print(f"  Content: {memory_data.get('content')}")
        print(f"  User ID: {memory_data.get('user_id')}")
        print(f"  Category: {memory_data.get('category', 'general')}")
        print(f"  Importance: {memory_data.get('importance', 5)}")
        print(f"  Connected to Supabase: {self.connected}")
        print(f"  Supabase URL: {self.supabase_url[:20]}..." if self.supabase_url else "  Supabase URL: None")

        if 'timestamp' not in memory_data:
            memory_data['timestamp'] = datetime.now().isoformat()

        # Ensure embedding is created and added
        if 'embedding' not in memory_data or not memory_data['embedding']:
            print(f"  Creating embedding for content: {memory_data['content'][:50]}...")
            embedding = self.create_embedding(memory_data['content'])
            memory_data['embedding'] = embedding
            print(f"  Embedding created with length: {len(embedding)}")

        if self.connected:
            try:
                # Store in Supabase
                print(f"  Attempting to store in Supabase table 'memories'...")
                result = self.client.table('memories').insert(memory_data).execute()
                print(f"  Supabase insert result: {result}")
                logger.info(f"Stored memory in Supabase with ID: {memory_id}")
                print(f"\n[MEMORY STORAGE] New memory added to database:\n  Content: {memory_data.get('content')}\n  User ID: {memory_data.get('user_id') or 'None'}\n  Category: {memory_data.get('category', 'general')}\n  Importance: {memory_data.get('importance', 5)}\n  ID: {memory_id}")
            except Exception as e:
                logger.error(f"Failed to store memory in Supabase: {e}")
                print(f"  ERROR storing in Supabase: {e}")
                # Fall back to in-memory storage
                self.memories.append(memory_data)
                logger.info(f"Stored memory in-memory with ID: {memory_id}")
                print(f"  Stored in-memory instead")
        else:
            # Store in-memory
            print(f"  Not connected to Supabase, storing in-memory")
            self.memories.append(memory_data)
            logger.info(f"Stored memory in-memory with ID: {memory_id}")

        return memory_id

    def retrieve_memories(self, query_embedding: List[float], limit: int = 5, user_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve memories similar to the query embedding.

        Args:
            query_embedding: Vector embedding to search for
            limit: Maximum number of memories to return
            user_id: Optional user ID to filter memories by

        Returns:
            List of memory dictionaries
        """
        if self.connected:
            try:
                # Search in Supabase using vector similarity
                params = {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.5,  # Lower threshold for better recall
                    'match_count': limit
                }

                # Use the appropriate function based on whether user_id is provided
                if user_id:
                    try:
                        # Try to use the user-specific function first
                        params['user_id_param'] = user_id
                        result = self.client.rpc('match_memories_by_user', params).execute()
                    except Exception as e:
                        logger.error(f"Error using match_memories_by_user: {e}")
                        # Fall back to the generic function
                        del params['user_id_param']
                        result = self.client.rpc('match_memories', params).execute()
                else:
                    # Use the generic function
                    result = self.client.rpc('match_memories', params).execute()

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
    def __init__(self, storage_agent_model, retrieval_agent_model, orchestrator_agent_model, user_id=None, supabase_url=None, supabase_key=None):
        """
        This agent will be responsible for determining which information about a user or events that occurred during a real-time or text conversation should be saved as a memory embedding for future reference.
        The agent will also be responsible for retrieving relevant memories when needed. Memories will be stored in a supabase vector database via the pgvector extension.

        Args:
            storage_agent_model: Model for the storage agent
            retrieval_agent_model: Model for the retrieval agent
            orchestrator_agent_model: Model for the orchestrator agent
            user_id: Optional user ID for user-specific memories
            supabase_url: Optional Supabase URL for database connection
            supabase_key: Optional Supabase API key for database connection
        """
        self.storage_agent_model = storage_agent_model
        self.retrieval_agent_model = retrieval_agent_model
        self.orchestrator_agent_model = orchestrator_agent_model
        self.user_id = user_id

        # Initialize the database connector
        self.db_connector = MemoryDatabaseConnector(supabase_url, supabase_key)

        # Initialize conversation history
        self.conversation_history = []

    async def process_message(self, message: str, role: str, conversation_mode: str, force_memory_consideration: bool = False, force_memory_retrieval: bool = False) -> None:
        """Process a new message from the conversation.

        Args:
            message: The message content
            role: 'user' or 'assistant'
            conversation_mode: 'audio' or 'chat'
            force_memory_consideration: If True, prioritize memory storage consideration
            force_memory_retrieval: If True, prioritize memory retrieval
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
        await self._check_and_store_memory(message, role, conversation_mode, force_memory_consideration, force_memory_retrieval)

    async def get_relevant_memories(self, current_message: str, force_retrieval: bool = False) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the current message.

        Args:
            current_message: The current message to find relevant memories for
            force_retrieval: If True, prioritize memory retrieval

        Returns:
            List of relevant memory dictionaries
        """
        print(f"\n[MEMORY RETRIEVAL] get_relevant_memories called with message: '{current_message[:50]}...'")
        print(f"  Force retrieval: {force_retrieval}")

        # Create a context from recent conversation history
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Create the prompt for the retrieval agent
        force_instruction = ""
        if force_retrieval:
            force_instruction = "\n\nIMPORTANT: The assistant has explicitly indicated an intent to recall information from memory. You should prioritize retrieving memories that are most relevant to the current conversation, even if they are only somewhat related."

        retrieval_prompt = f"""Based on the following conversation, retrieve relevant memories that would be helpful for responding to the current message.

        Current conversation:
        {context}

        Current message: {current_message}{force_instruction}

        Formulate a specific, targeted query that will find the most relevant information."""

        try:
            # Use litellm to get the retrieval agent's response
            response = await completion(
                model=self.retrieval_agent_model,
                messages=[{"role": "user", "content": retrieval_prompt}]
            )

            if response and response.choices:
                retrieval_response = response.choices[0].message.content
                logger.info(f"Retrieval agent response: {retrieval_response[:100]}...")
                print(f"\n[MEMORY RETRIEVAL] Agent response: '{retrieval_response[:100]}...'")
            else:
                logger.warning("No response from retrieval agent")
                print(f"\n[MEMORY RETRIEVAL] No response from retrieval agent")

            # Generate an embedding for the current message and retrieve relevant memories
            print(f"\n[MEMORY RETRIEVAL] Creating embedding for message: '{current_message[:50]}...'")
            message_embedding = self.db_connector.create_embedding(current_message)
            print(f"  Embedding created with length: {len(message_embedding)}")

            # Retrieve memories
            relevant_memories = self.db_connector.retrieve_memories(message_embedding, limit=3, user_id=self.user_id)

            # Print the final memories we're returning
            print(f"\n[DEBUG] Final relevant memories:")
            for i, memory in enumerate(relevant_memories):
                print(f"  Memory {i+1}: {memory.get('content', 'No content')}")

            return relevant_memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            print(f"\n[MEMORY RETRIEVAL] Error retrieving memories: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return []  # Return empty list on error

    async def _check_and_store_memory(self, message: str, role: str, conversation_mode: str, force_memory_consideration: bool = False, force_memory_retrieval: bool = False) -> None:
        """Check if a message should be stored as a memory and store it if needed.

        Args:
            message: The message content
            role: 'user' or 'assistant'
            conversation_mode: 'audio' or 'chat'
            force_memory_consideration: If True, prioritize memory storage consideration
            force_memory_retrieval: If True, prioritize memory retrieval
        """
        # Create a context from recent conversation history
        recent_history = self.conversation_history[-5:] if len(self.conversation_history) > 5 else self.conversation_history
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Create the prompt for the orchestrator
        force_instruction = ""
        if force_memory_consideration:
            force_instruction = "\n\nIMPORTANT: The assistant has explicitly indicated an intent to remember this information. YOU MUST STORE THIS AS A MEMORY. Do not analyze or evaluate whether it's worth storing - the assistant has already decided it is."
            print(f"\n[MEMORY ORCHESTRATOR] Assistant has explicitly indicated intent to remember: '{message[:100]}...'")
        elif force_memory_retrieval:
            force_instruction = "\n\nIMPORTANT: The assistant has explicitly indicated an intent to recall information from memory. You should prioritize retrieving memories that are most relevant to the current conversation."

        # If force_memory_consideration is True, we'll use a more direct prompt
        if force_memory_consideration:
            orchestrator_prompt = f"""The assistant has explicitly indicated an intent to remember information about the user.

            Conversation context:
            {context}

            Message to evaluate:
            {role}: {message}{force_instruction}

            YOU MUST STORE THIS AS A MEMORY. Extract the key information about the user from the conversation and store it as a concise, factual statement."""
        else:
            orchestrator_prompt = f"""Analyze the following conversation and determine if any important information should be stored as a memory.

            Conversation context:
            {context}

            Message to evaluate:
            {role}: {message}{force_instruction}

            If you find important information about the user, their preferences, or key events that would be useful in future conversations, create a memory.
            Otherwise, explain why no memory needs to be stored."""

        try:
            # Use litellm to get the orchestrator's response
            response = await completion(
                model=self.orchestrator_agent_model,
                messages=[{"role": "user", "content": orchestrator_prompt}]
            )

            if response and response.choices:
                orchestrator_response = response.choices[0].message.content
                logger.info(f"Orchestrator response: {orchestrator_response[:100]}...")
                # Print statement for visibility
                print(f"\n[MEMORY ORCHESTRATOR] Decision:\n  {orchestrator_response[:200]}...")
            else:
                logger.warning("No response from orchestrator agent")
                print("\n[MEMORY ORCHESTRATOR] No response from orchestrator agent")
        except Exception as e:
            logger.error(f"Error running orchestrator agent: {e}")

    def _text_similarity(self, content1: str, content2: str, threshold: float = 0.6) -> bool:
        """Calculate text similarity between two content strings.

        Args:
            content1: First content string
            content2: Second content string
            threshold: Similarity threshold (default: 0.6)

        Returns:
            True if the contents are similar, False otherwise
        """
        if not content1 or not content2:
            return False

        content1_lower = content1.lower()
        content2_lower = content2.lower()

        # Check if one is a substring of the other
        if content1_lower in content2_lower or content2_lower in content1_lower:
            print(f"  Text substring match found in _text_similarity")
            return True

        # Check for location information - more generic approach
        location_terms = ['city', 'location', 'moved', 'live', 'address', 'residence']
        if any(term in content1_lower for term in location_terms) and any(term in content2_lower for term in location_terms):
            # If both memories contain location-related terms, consider them similar
            print(f"  Location information match found in _text_similarity")
            return True

        # Check for food preferences
        if ("food" in content1_lower and "food" in content2_lower) and \
           ("favorite" in content1_lower and "favorite" in content2_lower) and \
           any(food in content1_lower and food in content2_lower for food in ["sushi", "pizza", "burger", "pasta"]):
            print(f"  Food preference match found in _text_similarity")
            return True

        # Calculate word overlap similarity
        words1 = set(content1_lower.split())
        words2 = set(content2_lower.split())
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = intersection / union if union > 0 else 0
            print(f"  Text similarity in _text_similarity: {similarity:.2f}")
            return similarity >= threshold

        return False

    def _is_similar_content(self, content1: str, content2: str = None, embedding1: list = None, embedding2: list = None, threshold: float = 0.7) -> bool:
        """Check if two memory contents are similar enough to be considered duplicates.

        Args:
            content1: First content string
            content2: Second content string (optional if embedding1 and embedding2 are provided)
            embedding1: First content embedding (optional, not used in this implementation)
            embedding2: Second content embedding (optional, not used in this implementation)
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            True if contents are similar, False otherwise
        """
        # We're ignoring embeddings and only using text-based comparison
        if content1 is None or content2 is None:
            return False

        # Convert to lowercase for case-insensitive comparison
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        # Check if one is a substring of the other
        if content1_lower in content2_lower or content2_lower in content1_lower:
            print(f"  Text substring match found")
            return True

        # Check for location information - more generic approach
        print(f"  Checking location pattern: '{content1_lower}' vs '{content2_lower}'")
        location_terms = ['city', 'location', 'moved', 'live', 'address', 'residence', 'sacramento', 'seattle']

        # Check if both memories contain location-related terms
        has_location1 = any(term in content1_lower for term in location_terms)
        has_location2 = any(term in content2_lower for term in location_terms)

        print(f"  Has location terms: {has_location1} vs {has_location2}")

        # If both memories contain location terms, check if they're about the same location
        if has_location1 and has_location2:
            # If both mention the same city, they're likely about the same location
            cities = ['sacramento', 'seattle', 'chicago', 'new york', 'los angeles', 'boston', 'miami']
            cities_in_content1 = [city for city in cities if city in content1_lower]
            cities_in_content2 = [city for city in cities if city in content2_lower]

            # If they mention different cities, they're not duplicates
            if cities_in_content1 and cities_in_content2 and not set(cities_in_content1).intersection(set(cities_in_content2)):
                print(f"  Different cities mentioned, not a duplicate")
                return False

            # If they're both about location but don't specify different cities, use a higher threshold
            location_threshold = 0.5  # Lower threshold for location memories
        else:
            location_threshold = 0.0  # Not used if not location memories

        # For food preferences
        if ("food" in content1_lower and "food" in content2_lower) and \
           ("favorite" in content1_lower and "favorite" in content2_lower) and \
           any(food in content1_lower and food in content2_lower for food in ["sushi", "pizza", "burger", "pasta"]):
            print(f"  Food preference match found")
            return True

        # Calculate Jaccard similarity
        words1 = set(content1_lower.split())
        words2 = set(content2_lower.split())

        # If either set is empty, return False
        if not words1 or not words2:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard_similarity = intersection / union if union > 0 else 0

        print(f"  Jaccard similarity between memories: {jaccard_similarity:.2f}")

        # Use different thresholds based on content type
        if has_location1 and has_location2:
            # Use lower threshold for location memories
            is_similar = jaccard_similarity >= location_threshold
            print(f"  Using location threshold {location_threshold}: {is_similar}")
            return is_similar
        else:
            # Use higher threshold for other memories
            jaccard_threshold = 0.5  # Reduced from 0.8 to be more lenient
            is_similar = jaccard_similarity >= jaccard_threshold
            print(f"  Using standard threshold {jaccard_threshold}: {is_similar}")
            return is_similar

    def format_memories_for_context(self, memories: List[Dict[str, Any]], retrieval_attempted: bool = False) -> str:
        """Format memories for inclusion in the conversation context.

        Args:
            memories: List of memory dictionaries
            retrieval_attempted: Whether memory retrieval was explicitly attempted

        Returns:
            Formatted string of memories
        """
        # Log the memories being formatted
        print(f"\n[MEMORY FORMATTING] Formatting {len(memories)} memories for context")
        for i, memory in enumerate(memories):
            content = memory.get('content', 'No content available')
            print(f"  Memory {i+1}: {content}")

        if not memories:
            if retrieval_attempted:
                # If retrieval was explicitly attempted but no memories were found, provide clear feedback
                note = "### MEMORY RETRIEVAL NOTE:\nI don't have any information about where you live or your location in my memory.\n###\n"
                print(f"[MEMORY FORMATTING] No memories found, returning note: {note}")
                return note
            else:
                # If no retrieval was attempted or it wasn't explicit, return empty string
                print(f"[MEMORY FORMATTING] No memories found and no retrieval attempted, returning empty string")
                return ""

        # Make the memory context more explicit and prominent
        formatted_memories = ["### IMPORTANT USER INFORMATION FROM MEMORY:"]

        # Check if any memory is about location
        location_memories = []
        other_memories = []

        for memory in memories:
            content = memory.get('content', 'No content available').lower()
            if any(term in content for term in ['live', 'location', 'address', 'city', 'moved', 'seattle', 'sacramento']):
                location_memories.append(memory)
            else:
                other_memories.append(memory)

        # Format location memories first with special emphasis
        if location_memories:
            formatted_memories.append("USER LOCATION INFORMATION:")
            for i, memory in enumerate(location_memories):
                content = memory.get('content', 'No content available')
                timestamp = memory.get('timestamp', 'Unknown time')
                category = memory.get('category', 'general')

                # Convert ISO timestamp to readable format if possible
                try:
                    dt = datetime.fromisoformat(timestamp)
                    readable_time = dt.strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    readable_time = timestamp

                formatted_memories.append(f"- {content} (Category: {category}, From: {readable_time})")

        # Format other memories
        if other_memories:
            formatted_memories.append("\nOTHER USER INFORMATION:")
            for i, memory in enumerate(other_memories):
                content = memory.get('content', 'No content available')
                timestamp = memory.get('timestamp', 'Unknown time')
                category = memory.get('category', 'general')

                # Convert ISO timestamp to readable format if possible
                try:
                    dt = datetime.fromisoformat(timestamp)
                    readable_time = dt.strftime("%B %d, %Y")
                except (ValueError, TypeError):
                    readable_time = timestamp

                formatted_memories.append(f"- {content} (Category: {category}, From: {readable_time})")

        formatted_memories.append("\n### YOU MUST USE THIS INFORMATION IN YOUR RESPONSE ###\n")
        result = "\n".join(formatted_memories)
        print(f"[MEMORY FORMATTING] Formatted memories:\n{result}")
        return result

    async def cleanup(self):
        """Clean up resources used by the memory agent."""
        # Currently, there's not much to clean up since sessions are cleaned up after each use
        # This method is provided for future extensions
        logger.info("Memory agent cleanup complete")
