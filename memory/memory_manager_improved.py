import os
import json
import logging
import spacy
import numpy as np
from pathlib import Path
from supabase import create_client
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedMemorySearch:
    """An improved memory search implementation that combines vector similarity with keyword matching."""

    def __init__(self, nlp=None):
        self.nlp = nlp
        self.category_weights = {
            "preference": 1.2,
            "health": 1.5,
            "personal_info": 1.3,
            "family": 1.2,
            "dietary": 1.4,
            "education": 1.1,
            "career": 1.1,
            "plan": 1.0
        }

    def initialize(self):
        """Initialize the NLP model if not provided."""
        if self.nlp is not None:
            return True

        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return False

    def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for the given text using spaCy."""
        if not self.nlp:
            logger.error("spaCy model not loaded")
            return []

        doc = self.nlp(text)
        return doc.vector.tolist()

    def search_memories(self, query: str, memories: List[Dict[str, Any]],
                        threshold: float = 0.5, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories similar to the query using a hybrid approach.

        Args:
            query: The search query
            memories: List of memory dictionaries
            threshold: Minimum similarity threshold
            limit: Maximum number of results to return

        Returns:
            List of memory dictionaries with similarity scores
        """
        try:
            # Parse the query to extract key information
            query_doc = self.nlp(query)

            # Extract query intent and category
            query_intent, query_category = self._analyze_query_intent(query_doc)
            logger.info(f"Query intent: {query_intent}, category: {query_category}")

            # Calculate vector similarity
            vector_results = self._vector_similarity_search(query, memories, threshold)

            # Calculate keyword similarity
            keyword_results = self._keyword_similarity_search(query_doc, memories)

            # Combine results with weighted scoring
            combined_results = self._combine_search_results(
                vector_results,
                keyword_results,
                query_category
            )

            # Sort by combined score and limit results
            combined_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            top_results = combined_results[:limit]

            if not top_results:
                logger.info("No similar memories found")
                return []

            logger.info(f"Found {len(top_results)} similar memories")
            return top_results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def _vector_similarity_search(self, query: str, memories: List[Dict[str, Any]],
                                 threshold: float) -> List[Dict[str, Any]]:
        """Search for memories using vector similarity."""
        # Create embedding for the query
        query_embedding = self.create_embedding(query)

        # Calculate similarity for each memory
        results = []
        for memory in memories:
            # Handle different embedding formats
            memory_embedding = memory.get('embedding', [])

            # Convert embedding to list if it's not already
            if isinstance(memory_embedding, str):
                try:
                    # Try to parse as JSON
                    import json
                    memory_embedding = json.loads(memory_embedding)
                except:
                    # If not JSON, try to parse as string representation of array
                    try:
                        memory_embedding = memory_embedding.strip('[]').split(',')
                        memory_embedding = [float(x.strip()) for x in memory_embedding if x.strip()]
                    except:
                        logger.warning(f"Could not parse embedding for memory: {memory.get('id')}")
                        continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, memory_embedding)

            if similarity > threshold:
                # Add similarity score to memory
                memory_with_score = memory.copy()
                memory_with_score['vector_similarity'] = similarity
                results.append(memory_with_score)

        return results

    def _keyword_similarity_search(self, query_doc, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search for memories using keyword matching."""
        results = []

        for memory in memories:
            memory_doc = self.nlp(memory.get('content', ''))

            # Calculate keyword overlap
            keyword_score = self._calculate_keyword_score(query_doc, memory_doc)

            # Add keyword score to memory
            memory_with_score = memory.copy()
            memory_with_score['keyword_similarity'] = keyword_score
            results.append(memory_with_score)

        return results

    def _calculate_keyword_score(self, query_doc, memory_doc) -> float:
        """Calculate keyword similarity score between query and memory."""
        # Count exact token matches
        exact_matches = sum(1 for token in query_doc if token.text.lower() in memory_doc.text.lower())

        # Count lemma matches (handles different word forms)
        lemma_matches = sum(1 for token in query_doc
                           if any(token.lemma_ == t.lemma_ for t in memory_doc))

        # Count entity matches
        query_entities = set(ent.text.lower() for ent in query_doc.ents)
        memory_entities = set(ent.text.lower() for ent in memory_doc.ents)
        entity_matches = len(query_entities.intersection(memory_entities))

        # Calculate combined score
        if len(query_doc) == 0:
            return 0

        exact_score = exact_matches / len(query_doc) if len(query_doc) > 0 else 0
        lemma_score = lemma_matches / len(query_doc) if len(query_doc) > 0 else 0
        entity_score = entity_matches * 2  # Give more weight to entity matches

        return (exact_score + lemma_score + entity_score) / 3

    def _combine_search_results(self, vector_results: List[Dict[str, Any]],
                               keyword_results: List[Dict[str, Any]],
                               query_category: str) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results with category boosting."""
        # Create a mapping of memory IDs to keyword scores
        keyword_scores = {mem.get('id'): mem.get('keyword_similarity', 0) for mem in keyword_results}

        # Combine scores
        combined_results = []
        for memory in vector_results:
            memory_id = memory.get('id')
            vector_score = memory.get('vector_similarity', 0)
            keyword_score = keyword_scores.get(memory_id, 0)

            # Apply category boosting
            category_boost = self._calculate_category_boost(memory, query_category)

            # Calculate combined score (weighted average)
            combined_score = (vector_score * 0.6) + (keyword_score * 0.3) + (category_boost * 0.1)

            # Add scores to memory
            memory_with_scores = memory.copy()
            memory_with_scores['keyword_similarity'] = keyword_score
            memory_with_scores['category_boost'] = category_boost
            memory_with_scores['combined_score'] = combined_score
            memory_with_scores['similarity'] = combined_score  # For compatibility with existing code

            combined_results.append(memory_with_scores)

        return combined_results

    def _calculate_category_boost(self, memory: Dict[str, Any], query_category: str) -> float:
        """Calculate category boost based on memory category and query category."""
        memory_category = memory.get('category', '').lower()

        # Direct category match
        if memory_category == query_category:
            return 1.0

        # Related categories
        related_categories = {
            'health': ['dietary'],
            'dietary': ['health'],
            'preference': ['personal_info'],
            'personal_info': ['preference'],
            'education': ['career'],
            'career': ['education']
        }

        if query_category in related_categories and memory_category in related_categories.get(query_category, []):
            return 0.5

        # Apply category importance weight
        return self.category_weights.get(memory_category, 0.1) * 0.2

    def _analyze_query_intent(self, query_doc) -> Tuple[str, str]:
        """Analyze query to determine intent and relevant category."""
        # Extract key nouns and verbs
        key_terms = [token.text.lower() for token in query_doc
                    if token.pos_ in ('NOUN', 'VERB') and not token.is_stop]

        # Map common terms to categories
        category_terms = {
            'health': ['health', 'medical', 'allergy', 'allergic', 'sick', 'illness'],
            'dietary': ['food', 'eat', 'diet', 'vegetarian', 'vegan', 'meal', 'dinner'],
            'preference': ['like', 'enjoy', 'prefer', 'favorite', 'hobby', 'interest'],
            'personal_info': ['pet', 'dog', 'cat', 'animal', 'live', 'address'],
            'family': ['family', 'sister', 'brother', 'parent', 'mother', 'father', 'relative'],
            'education': ['school', 'college', 'university', 'degree', 'graduate', 'study'],
            'career': ['job', 'work', 'career', 'profession', 'company', 'office'],
            'plan': ['plan', 'trip', 'travel', 'vacation', 'visit', 'future']
        }

        # Count category matches
        category_counts = {category: 0 for category in category_terms}
        for term in key_terms:
            for category, terms in category_terms.items():
                if term in terms or any(term in t for t in terms):
                    category_counts[category] += 1

        # Determine primary category
        if any(count > 0 for count in category_counts.values()):
            primary_category = max(category_counts.items(), key=lambda x: x[1])[0]
        else:
            primary_category = 'general'

        # Determine intent
        intent_terms = {
            'preference': ['like', 'enjoy', 'prefer', 'favorite'],
            'information': ['what', 'tell', 'know', 'information'],
            'confirmation': ['does', 'is', 'has', 'have'],
            'location': ['where', 'location', 'place'],
            'time': ['when', 'time', 'date']
        }

        intent_counts = {intent: 0 for intent in intent_terms}
        for token in query_doc:
            for intent, terms in intent_terms.items():
                if token.text.lower() in terms:
                    intent_counts[intent] += 1

        if any(count > 0 for count in intent_counts.values()):
            primary_intent = max(intent_counts.items(), key=lambda x: x[1])[0]
        else:
            primary_intent = 'general'

        return primary_intent, primary_category

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0

        try:
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)

            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)

            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0

            return dot_product / (norm_vec1 * norm_vec2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0


class ImprovedMemoryManager:
    """An improved memory manager with enhanced search capabilities."""

    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.supabase_client = None
        self.nlp = None
        self.search_engine = None

    def load_credentials(self):
        """Load Supabase credentials from .api_key.json file."""
        try:
            api_key_path = Path('.api_key.json')
            with open(api_key_path, 'r') as f:
                api_keys = json.load(f)
                self.supabase_url = api_keys.get('supabase_url')
                self.supabase_key = api_keys.get('supabase_api_key')

            if not self.supabase_url or not self.supabase_key:
                logger.error("Supabase credentials not found or incomplete in .api_key.json")
                return False

            logger.info("Supabase credentials loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Supabase credentials: {e}")
            return False

    def connect(self):
        """Connect to Supabase and load spaCy model."""
        if not self.load_credentials():
            return False

        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Connected to Supabase successfully")

            # Load spaCy model
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model successfully")

            # Initialize search engine
            self.search_engine = ImprovedMemorySearch(self.nlp)

            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def create_embedding(self, text):
        """Create an embedding for the given text using spaCy."""
        if not self.nlp:
            logger.error("spaCy model not loaded")
            return None

        doc = self.nlp(text)
        return doc.vector.tolist()

    def list_memories(self, limit=100, user_id=None):
        """List memories from the database.

        Args:
            limit: Maximum number of memories to return
            user_id: Optional user ID to filter memories by

        Returns:
            List of memory dictionaries
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return []

        try:
            # Build the query
            query = self.supabase_client.table('memories').select('*')

            # Filter by user_id if provided
            if user_id:
                query = query.eq('user_id', user_id)

            # Execute the query with ordering and limit
            response = query.order('timestamp', desc=True).limit(limit).execute()

            if not response.data:
                logger.info("No memories found")
                return []

            logger.info(f"Retrieved {len(response.data)} memories")
            return response.data
        except Exception as e:
            logger.error(f"Error listing memories: {e}")
            return []

    def add_memory(self, content, importance, category, source="manual", conversation_mode="manual", user_id=None):
        """Add a new memory to the database.

        Args:
            content: The content of the memory
            importance: Importance rating (1-10)
            category: Category of the memory
            source: Source of the memory
            conversation_mode: Mode of conversation
            user_id: Optional user ID to associate with the memory

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            # Create embedding
            embedding = self.create_embedding(content)

            # Create memory data
            memory_id = str(time.time())
            memory_data = {
                'id': memory_id,
                'content': content,
                'embedding': embedding,
                'importance': importance,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'conversation_mode': conversation_mode,
                'user_id': user_id
            }

            # Store in Supabase
            self.supabase_client.table('memories').insert(memory_data).execute()
            logger.info(f"Memory added with ID: {memory_id}")

            # Print statement for visibility
            print(f"\n[MEMORY STORAGE] New memory added to database:\n  Content: {content}\n  User ID: {user_id or 'None'}\n  Category: {category}\n  Importance: {importance}\n  ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None

    def search_memories(self, query, threshold=0.5, limit=5, user_id=None):
        """Search for memories similar to the query using improved search.

        Args:
            query: The search query
            threshold: Minimum similarity threshold
            limit: Maximum number of results to return
            user_id: Optional user ID to filter memories by

        Returns:
            List of memory dictionaries with similarity scores
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return []

        try:
            # Get all memories for local processing, filtered by user_id if provided
            all_memories = self.list_memories(limit=100, user_id=user_id)  # Get a larger set for processing

            # Ensure search engine is initialized
            if not self.search_engine:
                self.search_engine = ImprovedMemorySearch(self.nlp)

            # Search using hybrid approach
            results = self.search_engine.search_memories(query, all_memories, threshold, limit)

            if not results:
                logger.info("No similar memories found")
                return []

            logger.info(f"Found {len(results)} similar memories")
            return results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def search_memories_legacy(self, query, limit=5, user_id=None):
        """Search for memories using the original vector similarity approach.

        Args:
            query: The search query
            limit: Maximum number of results to return
            user_id: Optional user ID to filter memories by

        Returns:
            List of memory dictionaries with similarity scores
        """
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return []

        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)

            # Search using vector similarity
            try:
                # Include user_id in the RPC call if provided
                params = {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.5,
                    'match_count': limit
                }

                # Use the appropriate function based on whether user_id is provided
                if user_id:
                    try:
                        # Try to use the user-specific function first
                        params['user_id_param'] = user_id
                        result = self.supabase_client.rpc('match_memories_by_user', params).execute()
                        logger.info(f"Using match_memories_by_user with user_id: {user_id}")
                    except Exception as e:
                        logger.error(f"Error using match_memories_by_user: {e}")
                        # Fall back to the generic function
                        del params['user_id_param']
                        result = self.supabase_client.rpc('match_memories', params).execute()
                        logger.info("Falling back to match_memories without user_id")
                else:
                    # Use the generic function
                    result = self.supabase_client.rpc('match_memories', params).execute()

                if not result.data:
                    logger.info("No similar memories found")
                    return []

                logger.info(f"Found {len(result.data)} similar memories")
                return result.data
            except Exception as e:
                logger.warning(f"Error with RPC call: {e}")
                logger.info("Falling back to improved search")
                return self.search_memories(query, 0.5, limit)
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id):
        """Delete a memory from the database."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            self.supabase_client.table('memories').delete().eq('id', memory_id).execute()
            logger.info(f"Deleted memory with ID: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False

    def clear_all_memories(self):
        """Clear all memories from the database."""
        if not self.supabase_client:
            logger.error("Not connected to Supabase")
            return False

        try:
            self.supabase_client.rpc('clear_all_memories').execute()
            logger.info("All memories cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False


# Example usage
def test_improved_memory_manager():
    """Test the improved memory manager."""
    # Initialize memory manager
    manager = ImprovedMemoryManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
        return False

    # Add test memories
    test_memories = [
        {
            "content": "User enjoys hiking in the mountains on weekends",
            "importance": 8,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User's favorite color is blue",
            "importance": 5,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is allergic to peanuts and has a severe reaction",
            "importance": 10,
            "category": "health",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User has a dog named Max that is a golden retriever",
            "importance": 7,
            "category": "personal_info",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User prefers to read science fiction books, especially by Isaac Asimov",
            "importance": 6,
            "category": "preference",
            "source": "test",
            "conversation_mode": "chat"
        }
    ]

    # Add memories
    memory_ids = []
    for memory in test_memories:
        memory_id = manager.add_memory(
            content=memory["content"],
            importance=memory["importance"],
            category=memory["category"],
            source=memory["source"],
            conversation_mode=memory["conversation_mode"]
        )
        if memory_id:
            memory_ids.append(memory_id)
            logger.info(f"Added test memory: {memory['content']}")

    # Test search
    test_queries = [
        "What does the user like to do outdoors?",
        "Does the user have any pets?",
        "What should I know about the user's health?",
        "What books does the user enjoy reading?"
    ]

    for query in test_queries:
        logger.info(f"\nSearching for: {query}")

        # Try legacy search first
        try:
            logger.info("Legacy search results:")
            legacy_results = manager.search_memories_legacy(query)

            if legacy_results:
                for i, memory in enumerate(legacy_results):
                    similarity = memory.get('similarity', 0)
                    logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
            else:
                logger.info("No relevant memories found with legacy search")
        except Exception as e:
            logger.error(f"Error with legacy search: {e}")

        # Try improved search
        logger.info("\nImproved search results:")
        improved_results = manager.search_memories(query)

        if improved_results:
            for i, memory in enumerate(improved_results):
                vector_sim = memory.get('vector_similarity', 0)
                keyword_sim = memory.get('keyword_similarity', 0)
                category_boost = memory.get('category_boost', 0)
                combined_score = memory.get('combined_score', 0)

                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')}")
                logger.info(f"   Vector: {vector_sim:.2f}, Keyword: {keyword_sim:.2f}, Category: {category_boost:.2f}, Combined: {combined_score:.2f}")
        else:
            logger.info("No relevant memories found with improved search")

    # Clean up test memories
    logger.info("\nCleaning up test memories")
    for memory_id in memory_ids:
        manager.delete_memory(memory_id)
    logger.info(f"Deleted {len(memory_ids)} test memories")

    return True

if __name__ == "__main__":
    test_improved_memory_manager()
