import os
import json
import logging
import asyncio
from pathlib import Path
from memory_utils import MemoryManager
import spacy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_search():
    """Test the memory search functionality with various queries and thresholds."""
    
    # Initialize memory manager
    manager = MemoryManager()
    if not manager.connect():
        logger.error("Failed to connect to Supabase")
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
        # Modify the search_memories method to use a lower threshold
        original_threshold = 0.5
        manager.supabase_client.rpc = lambda name, params: manager.supabase_client.rpc(
            name,
            {**params, 'match_threshold': 0.3}
        ) if name == 'match_memories' else manager.supabase_client.rpc(name, params)
        
        results = manager.search_memories(query)
        
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
        results = manager.search_memories(query)
        
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
        keyword_results = manager.search_memories(keyword_query)
        
        if keyword_results:
            logger.info(f"Found {len(keyword_results)} relevant memories:")
            for i, memory in enumerate(keyword_results):
                similarity = memory.get('similarity', 0)
                logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
        else:
            logger.info("No relevant memories found")
        
        logger.info(f"\nSemantic query: {semantic_query}")
        semantic_results = manager.search_memories(semantic_query)
        
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
        vector_results = manager.search_memories(query)
        
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
