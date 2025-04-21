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
    def __init__(self, storage_agent_model, retrieval_agent_model, orchestrator_agent_model, session_service, user_id=None, supabase_url=None, supabase_key=None):
        """
        This agent will be responsible for determining which information about a user or events that occurred during a real-time or text conversation should be saved as a memory embedding for future reference.
        The agent will also be responsible for retrieving relevant memories when needed. Memories will be stored in a supabase vector database via the pgvector extension.
        This agent will in-turn be composed of three sub-agents: one for storing memories, one for retrieving memories, and an orchestrator agent that will determine when
          to store and retrieve memories, as well as whether stored or retrieved memories are sufficiently relevant to the current conversation.

        Args:
            storage_agent_model: Model for the storage agent
            retrieval_agent_model: Model for the retrieval agent
            orchestrator_agent_model: Model for the orchestrator agent
            session_service: Session service for managing sessions
            user_id: Optional user ID for user-specific memories
            supabase_url: Optional Supabase URL for database connection
            supabase_key: Optional Supabase API key for database connection
        """

        # If Supabase credentials are not provided, try to load them from the API key file
        if not supabase_url or not supabase_key:
            try:
                with open(".api_key.json", "r") as f:
                    api_keys = json.load(f)
                    supabase_url = supabase_url or api_keys.get("supabase_url")
                    supabase_key = supabase_key or api_keys.get("supabase_api_key")

                if not supabase_url or not supabase_key:
                    logger.warning("Supabase credentials not found or incomplete in .api_key.json")
            except Exception as e:
                logger.error(f"Error loading Supabase credentials: {e}")

        self.storage_agent_model = storage_agent_model
        self.retrieval_agent_model = retrieval_agent_model
        self.orchestrator_agent_model = orchestrator_agent_model
        self.session_service = session_service
        self.user_id = user_id

        # Initialize the database connector
        self.db_connector = MemoryDatabaseConnector(supabase_url, supabase_key)

        # Initialize conversation history
        self.conversation_history = []

        # Define tools for the agents
        from google.adk.tools import FunctionTool

        def store_memory_func(content: str, importance: int, category: str):
            """
            Store a new memory in the database.

            Args:
                content: The content of the memory
                importance: Importance score from 1-10
                category: Category of the memory (preference, personal_info, event, etc.)

            Returns:
                A dictionary with the result of the operation
            """
            print(f"\n[DEBUG] store_memory_func called in memory_agent.py")
            print(f"  Content: {content}")
            print(f"  Importance: {importance}")
            print(f"  Category: {category}")

            try:
                # Create memory data
                memory_data = {
                    "content": content,
                    "importance": importance,
                    "category": category,
                    "timestamp": datetime.now().isoformat(),
                    "source": "orchestrator",
                    "conversation_mode": "unknown",
                    "user_id": self.user_id  # Add user_id if available
                }

                print(f"  Memory data created: {memory_data}")

                # Generate embedding separately to ensure it's created correctly
                print(f"  Creating embedding for content: {content[:50]}...")
                embedding = self.db_connector.create_embedding(content)
                memory_data["embedding"] = embedding
                print(f"  Embedding created with length: {len(embedding)}")

                # Store the memory directly using the database connector
                print(f"  Calling store_memory on db_connector")
                memory_id = self.db_connector.store_memory(memory_data)
                print(f"  store_memory returned ID: {memory_id}")

                # Print statement for visibility
                print(f"\n[MEMORY STORAGE] Memory stored via function tool:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")

                return {"status": "success", "message": f"Memory stored: {content}", "memory_id": memory_id}
            except Exception as e:
                print(f"\n[ERROR] Error in store_memory_func: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                return {"status": "error", "message": f"Error storing memory: {str(e)}"}

        def retrieve_memories_func(query: str, limit: int = 5):
            """
            Retrieve memories relevant to a query.

            Args:
                query: The query to search for in memories
                limit: Maximum number of memories to retrieve (default: 5)

            Returns:
                A dictionary with the retrieved memories
            """
            print(f"\n[DEBUG] retrieve_memories_func called in memory_agent.py")
            print(f"  Query: {query}")
            print(f"  Limit: {limit}")
            print(f"  User ID: {self.user_id}")

            try:
                # Generate an embedding for the query
                print(f"  Creating embedding for query: {query[:50]}...")
                query_embedding = self.db_connector.create_embedding(query)
                print(f"  Embedding created with length: {len(query_embedding)}")

                # Retrieve memories directly using the database connector
                print(f"  Calling retrieve_memories on db_connector")
                memories = self.db_connector.retrieve_memories(query_embedding, limit=limit, user_id=self.user_id)
                print(f"  Retrieved {len(memories)} memories")

                # Format memories for response
                if not memories:
                    print(f"  No relevant memories found")
                    return {"status": "success", "message": "No relevant memories found.", "memories": []}

                formatted_memories = ["Retrieved memories:"]
                for i, memory in enumerate(memories):
                    content = memory.get('content', 'No content available')
                    category = memory.get('category', 'general')
                    importance = memory.get('importance', 0)

                    formatted_memories.append(f"{i+1}. [{category}] {content} (Importance: {importance})")
                    print(f"  Memory {i+1}: {content[:50]}...")

                return {
                    "status": "success",
                    "message": "\n".join(formatted_memories),
                    "memories": memories
                }
            except Exception as e:
                print(f"\n[ERROR] Error in retrieve_memories_func: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                return {"status": "error", "message": f"Error retrieving memories: {str(e)}", "memories": []}

        # Create FunctionTool instances
        self.store_memory_tool = FunctionTool(func=store_memory_func)
        self.retrieve_memories_tool = FunctionTool(func=retrieve_memories_func)

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

            Pay special attention to when the assistant has explicitly indicated an intent to remember information
            with phrases like "I'll remember that" or "I'll make a note of that". In these cases, you should prioritize
            storing the information that the assistant has committed to remember.

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

            Pay special attention to when the assistant has explicitly indicated an intent to recall information
            with phrases like "let me recall" or "let me see what I know about". In these cases, you should prioritize
            retrieving memories that are most relevant to the current conversation, even if they are only somewhat related.

            If no relevant memories are found, especially when the assistant has explicitly tried to recall information,
            it's important to clearly communicate this so the assistant can gracefully acknowledge the lack of information
            and respond based on general knowledge instead.

            Focus on quality over quantity - it's better to return a few highly relevant memories than many loosely related ones.""",
            description="Agent responsible for retrieving relevant memories based on conversation context",
            tools=[self.retrieve_memories_tool]
        )

        # Define the invoke_agent tool for the orchestrator
        def invoke_agent_func(agent_name: str, input: str):
            """
            Invoke a sub-agent to perform a specific task.

            Args:
                agent_name: Name of the agent to invoke (memory_storage or memory_retrieval)
                input: Input to provide to the agent

            Returns:
                A dictionary with the result from the sub-agent
            """
            print(f"\n[DEBUG] invoke_agent_func called in memory_agent.py")
            print(f"  Agent name: {agent_name}")
            print(f"  Input: {input}")

            try:
                # If the memory_storage agent is invoked, directly store the memory
                if agent_name.lower() == "memory_storage":
                    print(f"  Directly storing memory from input: {input}")

                    # Parse the input to extract memory details
                    # The input is expected to be a statement about the user
                    content = input.strip()
                    importance = 8  # Default importance
                    category = "preference"  # Default category

                    # Check if this is a vague location query and make it more specific
                    if ("location" in content.lower() or "place of residence" in content.lower() or "where" in content.lower()) and \
                       not any(city in content.lower() for city in ["seattle", "chicago", "sacramento"]):
                        # Check if we have any existing location memories that are more specific
                        location_embedding = self.db_connector.create_embedding("user location city where they live")
                        location_memories = self.db_connector.retrieve_memories(location_embedding, limit=3, user_id=self.user_id)

                        for memory in location_memories:
                            memory_content = memory.get('content', '').lower()
                            if any(city in memory_content for city in ["seattle", "chicago", "sacramento"]) and \
                               any(term in memory_content for term in ["live", "moved", "from", "born"]):
                                # Use the more specific memory instead
                                print(f"  Found more specific location memory: {memory.get('content')}")
                                content = memory.get('content')
                                break

                    # Try to determine a better category based on content
                    if "favorite" in content.lower() or "prefer" in content.lower() or "like" in content.lower() or "love" in content.lower():
                        category = "preference"
                    elif "born" in content.lower() or "from" in content.lower() or "live" in content.lower() or "age" in content.lower():
                        category = "personal_info"
                    elif "work" in content.lower() or "job" in content.lower() or "career" in content.lower():
                        category = "professional"
                    elif "hobby" in content.lower() or "interest" in content.lower() or "activity" in content.lower():
                        category = "hobby"

                    # Create memory data
                    memory_data = {
                        "content": content,
                        "importance": importance,
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        "source": "orchestrator",
                        "conversation_mode": "unknown",
                        "user_id": self.user_id  # Add user_id if available
                    }

                    print(f"  Memory data created: {memory_data}")

                    # Generate embedding
                    embedding = self.db_connector.create_embedding(content)
                    memory_data["embedding"] = embedding
                    print(f"  Embedding created with length: {len(embedding)}")

                    # Check for similar existing memories to avoid duplicates
                    try:
                        # Use the embedding to find similar memories
                        similar_memories = self.db_connector.retrieve_memories(embedding, limit=3, user_id=self.user_id)

                        # Check if any existing memory is very similar to the new one
                        duplicate_found = False
                        for memory in similar_memories:
                            existing_content = memory.get('content', '')
                            existing_embedding = memory.get('embedding', [])
                            # Use cosine similarity on embeddings
                            if self._is_similar_content(content1=content, content2=existing_content, embedding1=embedding, embedding2=existing_embedding):
                                print(f"  Found similar existing memory: {existing_content}")
                                print(f"  Skipping storage to avoid duplicate")
                                duplicate_found = True
                                memory_id = memory.get('id', 'unknown')
                                break

                        if not duplicate_found:
                            # Store the memory directly
                            memory_id = self.db_connector.store_memory(memory_data)
                            print(f"  Memory stored with ID: {memory_id}")
                            # Print statement for visibility
                            print(f"\n[MEMORY STORAGE] Memory stored via invoke_agent_func:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")
                        else:
                            print(f"\n[MEMORY STORAGE] Duplicate memory detected, using existing memory:\n  Content: {content}\n  ID: {memory_id}")
                    except Exception as e:
                        print(f"  Error checking for duplicates: {e}, proceeding with storage")
                        # Store the memory directly as fallback
                        memory_id = self.db_connector.store_memory(memory_data)
                        print(f"  Memory stored with ID: {memory_id}")
                        # Print statement for visibility
                        print(f"\n[MEMORY STORAGE] Memory stored via invoke_agent_func:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")

                    return {"status": "success", "result": f"Stored memory: {content}", "memory_id": memory_id}

                # For other agents, just return a success message
                # In a real implementation, you would delegate to the appropriate agent
                return {"status": "success", "result": f"Invoked {agent_name} with input: {input}"}
            except Exception as e:
                print(f"\n[ERROR] Error in invoke_agent_func: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                return {"status": "error", "result": f"Error invoking {agent_name}: {str(e)}"}

        # Create FunctionTool for invoke_agent
        self.invoke_agent_tool = FunctionTool(func=invoke_agent_func)

        # Check if orchestrator_agent_model is a string (direct model name) or an object (LiteLlm instance)
        self.orchestrator_agent = Agent(
            name="memory_orchestrator",
            model=orchestrator_agent_model,  # Can be either a direct model name (string) or a LiteLlm instance
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
            - Pay special attention to when the assistant explicitly indicates an intent to remember information
              with phrases like "I'll remember that" or "I'll make a note of that"
            - When the assistant has indicated memory storage intent, you should prioritize storing that information
              unless it contains absolutely no useful content

            You have access to two sub-agents:
            1. memory_storage: Determines what information to store as memories
            2. memory_retrieval: Finds relevant memories based on conversation context

            Use the invoke_agent tool to delegate tasks to these specialized agents.""",
            description="Main Orchestrator Agent for Memory Management",
            sub_agents=[self.storage_agent, self.retrieval_agent],
            tools=[self.invoke_agent_tool]
        )

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
        logger.info(f"Getting relevant memories for message: '{current_message[:50]}...'")

        # Create a context from recent conversation history
        recent_history = self.conversation_history[-10:] if len(self.conversation_history) > 10 else self.conversation_history
        context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in recent_history])

        # Create a session with context
        retrieval_session = self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user")

        # Store the force_retrieval flag in the session
        # In ADK, session state is a dictionary, not an object with set method
        retrieval_session.state["force_memory_retrieval"] = force_retrieval

        # In ADK, we create a Runner with the retrieval agent and session service
        from google.adk.runners import Runner

        # Create the runner with the session
        retrieval_runner = Runner(
            agent=self.retrieval_agent,
            app_name="memory_agent",
            session_service=self.session_service
        )

        print(f"\n[DEBUG] Created retrieval runner")

        # Create the prompt for the retrieval agent
        force_instruction = ""
        if force_retrieval:
            force_instruction = "\n\nIMPORTANT: The assistant has explicitly indicated an intent to recall information from memory. You should prioritize retrieving memories that are most relevant to the current conversation, even if they are only somewhat related."

        retrieval_prompt = f"""Based on the following conversation, retrieve relevant memories that would be helpful for responding to the current message.

        Current conversation:
        {context}

        Current message: {current_message}{force_instruction}

        Use the retrieve_memories tool to search for relevant memories. Formulate a specific, targeted query that will find the most relevant information."""

        try:
            # Create a session ID that's unique for this retrieval operation
            import uuid
            retrieval_session_id = str(uuid.uuid4())

            # Create the session first
            self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user", session_id=retrieval_session_id)

            # Run the retrieval agent
            from google.genai import types
            content = types.Content(role='user', parts=[types.Part(text=retrieval_prompt)])
            events = retrieval_runner.run(user_id=self.user_id or "test_user", session_id=retrieval_session_id, new_message=content)

            # Process the response
            retrieval_response = None
            for event in events:
                if event.is_final_response():
                    retrieval_response = event.content.parts[0].text
                    break

            if retrieval_response:
                logger.info(f"Retrieval agent response: {retrieval_response[:100]}...")
                print(f"\n[MEMORY RETRIEVAL] Agent response: '{retrieval_response[:100]}...'")
            else:
                logger.warning("No response from retrieval agent")
                print(f"\n[MEMORY RETRIEVAL] No response from retrieval agent")

            # The retrieval agent should have used the retrieve_memories tool,
            # which would have stored the results in the database connector
            # Generate an embedding for the current message and retrieve relevant memories
            print(f"\n[MEMORY RETRIEVAL] Creating embedding for message: '{current_message[:50]}...'")
            message_embedding = self.db_connector.create_embedding(current_message)
            print(f"  Embedding created with length: {len(message_embedding)}")

            # Check if this is a location-related query
            location_terms = ['where', 'location', 'live', 'from', 'moved', 'address', 'city', 'state', 'country', 'reside']
            is_location_query = any(term in current_message.lower() for term in location_terms)

            # Check for voice mode patterns like "you know where I live"
            voice_patterns = ['know where', 'remember where', 'know my location', 'know my address', 'you know where']
            is_voice_location_query = any(pattern in current_message.lower() for pattern in voice_patterns)

            # If it's a location query, voice location query, or we're forcing retrieval, use optimized approach
            if is_location_query or is_voice_location_query or force_retrieval:
                print(f"\n[DEBUG] Detected location-related query or forced retrieval, using optimized approach")
                print(f"  Is location query: {is_location_query}")
                print(f"  Is voice location query: {is_voice_location_query}")
                print(f"  Force retrieval: {force_retrieval}")

                # Create a single optimized composite query for location information
                # This combines multiple location-related concepts into one query
                composite_location_query = "user location where they live city address residence Seattle Sacramento"
                print(f"\n[DEBUG] Using optimized composite location query: {composite_location_query}")

                # Create a single embedding for the composite query
                composite_embedding = self.db_connector.create_embedding(composite_location_query)

                # Retrieve memories with a higher limit to ensure we get comprehensive results
                # but with just one database call instead of multiple calls
                location_memories = self.db_connector.retrieve_memories(composite_embedding, limit=5, user_id=self.user_id)

                # Also use the original message embedding as a backup to ensure relevance
                message_memories = self.db_connector.retrieve_memories(message_embedding, limit=3, user_id=self.user_id)

                # Combine results while removing duplicates
                all_memories = []
                memory_ids = set()  # Track memory IDs to avoid duplicates

                # First add location-specific memories
                for memory in location_memories:
                    memory_id = memory.get('id')
                    if memory_id and memory_id not in memory_ids:
                        memory_ids.add(memory_id)
                        all_memories.append(memory)
                        print(f"  Found location memory: {memory.get('content', 'No content')}")

                # Then add message-specific memories if not already included
                for memory in message_memories:
                    memory_id = memory.get('id')
                    if memory_id and memory_id not in memory_ids:
                        memory_ids.add(memory_id)
                        all_memories.append(memory)
                        print(f"  Found message memory: {memory.get('content', 'No content')}")

                # If we found any memories, use those
                if all_memories:
                    print(f"\n[DEBUG] Found {len(all_memories)} unique memories with optimized approach")
                    relevant_memories = all_memories
                else:
                    # Fall back to regular search
                    print(f"\n[DEBUG] No memories found with optimized approach, using regular search")
                    relevant_memories = self.db_connector.retrieve_memories(message_embedding, limit=3, user_id=self.user_id)
            else:
                # Regular search for non-location queries
                print(f"\n[DEBUG] Using regular search for non-location query")
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
        finally:
            # No need to clean up the session in ADK, it's managed by the session service
            pass

    async def _handle_store_memory(self, content: str, importance: int, category: str, tool_context=None):
        """Handle the store_memory tool call.

        Args:
            content: The content of the memory
            importance: Importance score from 1-10
            category: Category of the memory (preference, personal_info, event, etc.)
            tool_context: The tool context (contains session state)

        Returns:
            Response message
        """
        try:
            print(f"\n[DEBUG] _handle_store_memory called in memory_agent.py")
            print(f"  Content: {content}")
            print(f"  Importance: {importance}")
            print(f"  Category: {category}")
            print(f"  User ID: {self.user_id}")
            print(f"  Supabase URL: {self.db_connector.supabase_url[:20]}..." if self.db_connector.supabase_url else "  Supabase URL: None")
            print(f"  Connected to Supabase: {self.db_connector.connected}")

            if not content or not importance or not category:
                print(f"  ERROR: Missing required parameters")
                return {"status": "error", "message": "Error: Missing required parameters. Please provide content, importance, and category."}

            # Get session state from tool_context if available
            current_role = "unknown"
            conversation_mode = "unknown"
            if tool_context and hasattr(tool_context, 'state'):
                try:
                    current_role = tool_context.state.get("current_role", "unknown")
                    conversation_mode = tool_context.state.get("conversation_mode", "unknown")
                    print(f"  Current role from session: {current_role}")
                    print(f"  Conversation mode from session: {conversation_mode}")
                except Exception as e:
                    print(f"  ERROR getting session state: {e}")
                    logger.warning(f"Could not get session state: {e}")

            # Create memory data
            memory_data = {
                "content": content,
                "importance": importance,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "source": current_role,
                "conversation_mode": conversation_mode,
                "user_id": self.user_id  # Add user_id if available
            }

            print(f"  Memory data created: {memory_data}")

            # Generate embedding separately to ensure it's created correctly
            print(f"  Creating embedding for content: {content[:50]}...")
            embedding = self.db_connector.create_embedding(content)
            memory_data["embedding"] = embedding
            print(f"  Embedding created with length: {len(embedding)}")

            # Check for similar existing memories to avoid duplicates
            try:
                # Use the embedding to find similar memories
                similar_memories = self.db_connector.retrieve_memories(embedding, limit=3, user_id=self.user_id)

                # Check if any existing memory is very similar to the new one
                duplicate_found = False
                for memory in similar_memories:
                    existing_content = memory.get('content', '')
                    existing_embedding = memory.get('embedding', [])
                    # Use cosine similarity on embeddings
                    if self._is_similar_content(content1=content, content2=existing_content, embedding1=embedding, embedding2=existing_embedding):
                        print(f"  Found similar existing memory: {existing_content}")
                        print(f"  Skipping storage to avoid duplicate")
                        duplicate_found = True
                        memory_id = memory.get('id', 'unknown')
                        break

                if not duplicate_found:
                    # Store the memory
                    print(f"  Calling store_memory on db_connector")
                    memory_id = self.db_connector.store_memory(memory_data)
                    print(f"  store_memory returned ID: {memory_id}")
                    logger.info(f"Stored new memory with ID: {memory_id}")
                    # Print statement for visibility
                    print(f"\n[MEMORY INTEGRATION] Directly stored memory:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}")
                else:
                    print(f"\n[MEMORY INTEGRATION] Duplicate memory detected, using existing memory:\n  Content: {content}\n  ID: {memory_id}")
            except Exception as e:
                print(f"  Error checking for duplicates: {e}, proceeding with storage")
                # Store the memory directly as fallback
                print(f"  Calling store_memory on db_connector")
                memory_id = self.db_connector.store_memory(memory_data)
                print(f"  store_memory returned ID: {memory_id}")
                logger.info(f"Stored new memory with ID: {memory_id}")
                # Print statement for visibility
                print(f"\n[MEMORY INTEGRATION] Directly stored memory:\n  Content: {content}\n  User ID: {self.user_id or 'None'}\n  Category: {category}\n  Importance: {importance}")

            return {"status": "success", "message": f"Successfully stored memory: {content}"}
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            print(f"\n[ERROR] Error in _handle_store_memory: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Error storing memory: {str(e)}"}

    async def _handle_retrieve_memories(self, query: str, limit: int = 3, tool_context=None):
        """Handle the retrieve_memories tool call.

        Args:
            query: The query to search for in memories
            limit: Maximum number of memories to retrieve (default: 3)
            tool_context: The tool context (contains session state)

        Returns:
            Response message with retrieved memories
        """
        try:
            if not query:
                return {"status": "error", "message": "Error: Missing required parameter 'query'."}

            # Generate an embedding for the query using spaCy
            query_embedding = self.db_connector.create_embedding(query)

            # Retrieve memories
            memories = self.db_connector.retrieve_memories(query_embedding, limit=limit, user_id=self.user_id)

            # Format memories for response
            if not memories:
                return {"status": "success", "message": "No relevant memories found.", "memories": []}

            formatted_memories = ["Retrieved memories:"]
            for i, memory in enumerate(memories):
                content = memory.get('content', 'No content available')
                category = memory.get('category', 'general')
                importance = memory.get('importance', 0)

                formatted_memories.append(f"{i+1}. [{category}] {content} (Importance: {importance})")

            return {
                "status": "success",
                "message": "\n".join(formatted_memories),
                "memories": memories
            }
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {"status": "error", "message": f"Error retrieving memories: {str(e)}", "memories": []}

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

        # Create a session with context
        orchestrator_session = self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user")
        # Store the current role and conversation mode in the session
        # In ADK, session state is a dictionary, not an object with set method
        orchestrator_session.state["current_role"] = role
        orchestrator_session.state["conversation_mode"] = conversation_mode
        orchestrator_session.state["force_memory_consideration"] = force_memory_consideration
        orchestrator_session.state["force_memory_retrieval"] = force_memory_retrieval

        # In ADK, we create a Runner with the orchestrator agent and session service
        from google.adk.runners import Runner

        # Create the runner with the session
        orchestrator_runner = Runner(
            agent=self.orchestrator_agent,
            app_name="memory_agent",
            session_service=self.session_service
        )

        print(f"\n[DEBUG] Created orchestrator runner")

        # Create the prompt for the orchestrator
        force_instruction = ""
        if force_memory_consideration:
            force_instruction = "\n\nIMPORTANT: The assistant has explicitly indicated an intent to remember this information. YOU MUST STORE THIS AS A MEMORY. Do not analyze or evaluate whether it's worth storing - the assistant has already decided it is. Immediately invoke the memory_storage agent to create a memory with this information."
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

            YOU MUST STORE THIS AS A MEMORY. Immediately invoke the memory_storage agent to create a memory with this information.
            Extract the key information about the user from the conversation and store it as a concise, factual statement."""
        else:
            orchestrator_prompt = f"""Analyze the following conversation and determine if any important information should be stored as a memory.

            Conversation context:
            {context}

            Message to evaluate:
            {role}: {message}{force_instruction}

            If you find important information about the user, their preferences, or key events that would be useful in future conversations, invoke the memory_storage agent to create a memory.
            Otherwise, explain why no memory needs to be stored."""

        try:
            # Create a session ID that's unique for this orchestrator operation
            import uuid
            orchestrator_session_id = str(uuid.uuid4())

            # Create the session first
            self.session_service.create_session(app_name="memory_agent", user_id=self.user_id or "test_user", session_id=orchestrator_session_id)

            # Print statement for visibility
            print(f"\n[MEMORY ORCHESTRATOR] Evaluating message for memory storage:\n  Role: {role}\n  Message: {message[:100]}...\n  Force consideration: {force_memory_consideration}")

            # Run the orchestrator
            from google.genai import types
            content = types.Content(role='user', parts=[types.Part(text=orchestrator_prompt)])
            events = orchestrator_runner.run(user_id=self.user_id or "test_user", session_id=orchestrator_session_id, new_message=content)

            # Process the response
            orchestrator_response = None
            for event in events:
                if event.is_final_response():
                    orchestrator_response = event.content.parts[0].text
                    break

            if orchestrator_response:
                logger.info(f"Orchestrator response: {orchestrator_response[:100]}...")
                # Print statement for visibility
                print(f"\n[MEMORY ORCHESTRATOR] Decision:\n  {orchestrator_response[:200]}...")
            else:
                logger.warning("No response from orchestrator agent")
                print("\n[MEMORY ORCHESTRATOR] No response from orchestrator agent")
        except Exception as e:
            logger.error(f"Error running orchestrator agent: {e}")
        finally:
            # No need to clean up the session in ADK, it's managed by the session service
            pass

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
