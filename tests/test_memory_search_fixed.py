import os
import json
import logging
import asyncio
from pathlib import Path
import spacy
import numpy as np
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMemoryManager:
    """A simplified memory manager that doesn't rely on Supabase for testing."""
    
    def __init__(self):
        self.memories = []
        self.nlp = None
        
    def connect(self):
        """Load spaCy model."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return False
    
    def create_embedding(self, text):
        """Create an embedding for the given text using spaCy."""
        if not self.nlp:
            logger.error("spaCy model not loaded")
            return None
            
        doc = self.nlp(text)
        return doc.vector.tolist()
    
    def add_memory(self, content, importance, category, source="test", conversation_mode="chat"):
        """Add a new memory."""
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
                'conversation_mode': conversation_mode
            }
            
            # Store in memory
            self.memories.append(memory_data)
            logger.info(f"Memory added with ID: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return None
    
    def list_memories(self, limit=10):
        """List memories."""
        return self.memories[:limit]
    
    def search_memories(self, query, threshold=0.5, limit=5):
        """Search for memories similar to the query using cosine similarity."""
        try:
            # Create embedding for the query
            query_embedding = self.create_embedding(query)
            
            # Calculate similarity for each memory
            results = []
            for memory in self.memories:
                memory_embedding = memory.get('embedding', [])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                
                if similarity > threshold:
                    # Add similarity score to memory
                    memory_with_score = memory.copy()
                    memory_with_score['similarity'] = similarity
                    results.append(memory_with_score)
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            # Limit results
            results = results[:limit]
            
            if not results:
                logger.info("No similar memories found")
                return []
                
            logger.info(f"Found {len(results)} similar memories")
            return results
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def delete_memory(self, memory_id):
        """Delete a memory."""
        for i, memory in enumerate(self.memories):
            if memory.get('id') == memory_id:
                del self.memories[i]
                logger.info(f"Deleted memory with ID: {memory_id}")
                return True
        
        logger.warning(f"Memory with ID {memory_id} not found")
        return False

async def test_memory_search():
    """Test the memory search functionality with various queries and thresholds."""
    
    # Initialize memory manager
    manager = SimpleMemoryManager()
    if not manager.connect():
        logger.error("Failed to initialize memory manager")
        return False
    
    # Add diverse test memories
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
        },
        {
            "content": "User graduated from Stanford University with a degree in Computer Science",
            "importance": 7,
            "category": "education",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is planning a trip to Japan next summer",
            "importance": 6,
            "category": "plan",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User works as a software engineer at Google",
            "importance": 8,
            "category": "career",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User has a sister named Emma who lives in New York",
            "importance": 7,
            "category": "family",
            "source": "test",
            "conversation_mode": "chat"
        },
        {
            "content": "User is vegetarian and doesn't eat meat",
            "importance": 9,
            "category": "dietary",
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
    
    if not memory_ids:
        logger.error("Failed to add any test memories")
        return False
    
    # List memories
    memories = manager.list_memories()
    logger.info(f"Retrieved {len(memories)} memories")
    
    # Test different search queries with default threshold (0.5)
    logger.info("\n=== Testing with default threshold (0.5) ===")
    search_queries = [
        "What does the user like to do outdoors?",
        "What are the user's preferences?",
        "Does the user have any pets?",
        "What should I know about the user's health?",
        "Where did the user go to college?",
        "What is the user's job?",
        "What are the user's travel plans?",
        "What are the user's dietary restrictions?",
        "Who are the user's family members?",
        "What books does the user enjoy reading?"
    ]
    
    for query in search_queries:
        logger.info(f"\nSearching for: {query}")
        results = manager.search_memories(query)
        
        if results:
            logger.info(f"Found {len(results)} relevant memories:")
            for i, memory in enumerate(results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
    
    # Test with lower threshold for better recall
    logger.info("\n=== Testing with lower threshold (0.3) ===")
    for query in search_queries:
        logger.info(f"\nSearching for: {query}")
        results = manager.search_memories(query, threshold=0.3)
        
        if results:
            logger.info(f"Found {len(results)} relevant memories:")
            for i, memory in enumerate(results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
    
    # Test with ambiguous or indirect queries
    logger.info("\n=== Testing with ambiguous or indirect queries ===")
    ambiguous_queries = [
        "What activities does the user enjoy?",
        "Tell me about the user's pet",
        "What should I avoid serving the user for dinner?",
        "Where did the user study?",
        "What is the user planning for vacation?",
        "Who is in the user's family?",
        "What kind of literature does the user like?"
    ]
    
    for query in ambiguous_queries:
        logger.info(f"\nSearching for: {query}")
        results = manager.search_memories(query, threshold=0.3)
        
        if results:
            logger.info(f"Found {len(results)} relevant memories:")
            for i, memory in enumerate(results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
    
    # Test with direct keyword matches vs semantic similarity
    logger.info("\n=== Testing keyword matches vs semantic similarity ===")
    keyword_vs_semantic = [
        ("dog pet", "Does the user have any animals?"),
        ("allergy", "What health concerns does the user have?"),
        ("mountain hiking", "What outdoor activities does the user enjoy?"),
        ("Japan travel", "Where is the user planning to go on vacation?"),
        ("vegetarian", "What food preferences does the user have?")
    ]
    
    for keyword_query, semantic_query in keyword_vs_semantic:
        logger.info(f"\nKeyword query: {keyword_query}")
        keyword_results = manager.search_memories(keyword_query, threshold=0.3)
        
        if keyword_results:
            logger.info(f"Found {len(keyword_results)} relevant memories:")
            for i, memory in enumerate(keyword_results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
        
        logger.info(f"\nSemantic query: {semantic_query}")
        semantic_results = manager.search_memories(semantic_query, threshold=0.3)
        
        if semantic_results:
            logger.info(f"Found {len(semantic_results)} relevant memories:")
            for i, memory in enumerate(semantic_results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
    
    # Compare spaCy embeddings with a simple keyword matching approach
    logger.info("\n=== Comparing vector embeddings with keyword matching ===")
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
    
    for query in search_queries[:5]:  # Test with first 5 queries
        logger.info(f"\nQuery: {query}")
        
        # Get vector similarity results
        vector_results = manager.search_memories(query, threshold=0.3)
        
        # Simple keyword matching
        query_doc = nlp(query)
        keyword_results = []
        
        for memory in memories:
            memory_doc = nlp(memory.get('content', ''))
            # Count token overlap
            overlap = sum(1 for token in query_doc if token.text.lower() in memory_doc.text.lower())
            if overlap > 0:
                keyword_results.append({
                    **memory,
                    'similarity': overlap / len(query_doc)
                })
        
        # Sort by similarity
        keyword_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        keyword_results = keyword_results[:5]  # Top 5
        
        logger.info("Vector similarity results:")
        if vector_results:
            for i, memory in enumerate(vector_results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
        
        logger.info("\nKeyword matching results:")
        if keyword_results:
            for i, memory in enumerate(keyword_results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
    
    # Clean up test memories
    logger.info("\nCleaning up test memories")
    for memory_id in memory_ids:
        manager.delete_memory(memory_id)
    logger.info(f"Deleted {len(memory_ids)} test memories")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_search())
    if success:
        logger.info("Memory search test completed successfully")
    else:
        logger.error("Memory search test failed")
