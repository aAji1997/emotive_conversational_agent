import os
import json
import logging
import asyncio
from pathlib import Path
from memory_utils import MemoryManager
import spacy
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_memory_thresholds():
    """Test how different similarity thresholds affect memory retrieval."""
    
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
    
    # Define test queries
    test_queries = [
        "What does the user like to do outdoors?",
        "What are the user's dietary restrictions?",
        "Does the user have any pets?",
        "What is the user's profession?",
        "What are the user's travel plans?"
    ]
    
    # Define thresholds to test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Store results for analysis
    results = {}
    
    # Test each query with different thresholds
    for query in test_queries:
        logger.info(f"\n=== Testing query: {query} ===")
        query_results = {}
        
        for threshold in thresholds:
            logger.info(f"\nTesting with threshold: {threshold}")
            
            # Create embedding for the query
            query_embedding = manager.create_embedding(query)
            
            # Search using vector similarity with custom threshold
            try:
                result = manager.supabase_client.rpc(
                    'match_memories',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': threshold,
                        'match_count': 10  # Get more results to see the effect
                    }
                ).execute()
                
                memories = result.data
                
                if memories:
                    logger.info(f"Found {len(memories)} memories with threshold {threshold}:")
                    for i, memory in enumerate(memories):
                        similarity = memory.get('similarity', 0)
                        logger.info(f"{i+1}. [{memory.get('category', 'unknown')}] {memory.get('content')} (Similarity: {similarity:.2f})")
                    
                    # Store results for analysis
                    query_results[threshold] = {
                        'count': len(memories),
                        'memories': [
                            {
                                'content': memory.get('content', ''),
                                'similarity': memory.get('similarity', 0),
                                'category': memory.get('category', 'unknown')
                            }
                            for memory in memories
                        ]
                    }
                else:
                    logger.info(f"No memories found with threshold {threshold}")
                    query_results[threshold] = {'count': 0, 'memories': []}
            
            except Exception as e:
                logger.error(f"Error searching memories with threshold {threshold}: {e}")
                query_results[threshold] = {'count': 0, 'memories': []}
        
        results[query] = query_results
    
    # Analyze results
    logger.info("\n=== Analysis of Results ===")
    
    # Calculate average number of memories retrieved at each threshold
    avg_counts = {}
    for threshold in thresholds:
        counts = [results[query][threshold]['count'] for query in test_queries]
        avg_counts[threshold] = sum(counts) / len(counts)
        logger.info(f"Threshold {threshold}: Average {avg_counts[threshold]:.2f} memories retrieved")
    
    # Calculate precision at different thresholds
    # For this test, we'll consider a memory "relevant" if its similarity score is above 0.7
    precision = {}
    for threshold in thresholds:
        relevant_count = 0
        total_count = 0
        
        for query in test_queries:
            memories = results[query][threshold]['memories']
            for memory in memories:
                total_count += 1
                if memory['similarity'] >= 0.7:  # Consider this a "relevant" memory
                    relevant_count += 1
        
        precision[threshold] = relevant_count / total_count if total_count > 0 else 0
        logger.info(f"Threshold {threshold}: Precision {precision[threshold]:.2f}")
    
    # Clean up test memories
    logger.info("\nCleaning up test memories")
    for memory_id in memory_ids:
        manager.delete_memory(memory_id)
    logger.info(f"Deleted {len(memory_ids)} test memories")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_memory_thresholds())
    if success:
        logger.info("Memory threshold test completed successfully")
    else:
        logger.error("Memory threshold test failed")
