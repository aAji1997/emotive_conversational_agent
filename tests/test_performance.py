import asyncio
import time
import logging
from datetime import datetime
import sys
import json
import os
from pathlib import Path

# Get the absolute path to the project root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from memory.memory_agent import ConversationMemoryAgent
from memory.memory_integration import MemoryIntegration
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from litellm import completion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_operations': [],
            'api_calls': [],
            'error_count': 0
        }
    
    def add_metric(self, category, value):
        if category in self.metrics:
            self.metrics[category].append(value)
    
    def get_average(self, category):
        if not self.metrics[category]:
            return 0
        return sum(self.metrics[category]) / len(self.metrics[category])
    
    def get_summary(self):
        return {
            'avg_response_time': self.get_average('response_times'),
            'avg_memory_operation_time': self.get_average('memory_operations'),
            'avg_api_call_time': self.get_average('api_calls'),
            'total_errors': self.metrics['error_count']
        }

async def test_memory_agent_performance():
    """Test the performance of the memory agent with various operations."""
    metrics = PerformanceMetrics()
    
    # Load API key
    try:
        with open('.api_key.json', 'r') as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get('openai_api_key')
            
        if not openai_api_key:
            logger.error("OpenAI API key not found in .api_key.json")
            return
    except Exception as e:
        logger.error(f"Error loading API key: {e}")
        return
    
    # Set the API key in environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Initialize components with direct model names
    storage_model = "gpt-4o-mini"
    retrieval_model = "gpt-4o-mini"
    orchestrator_model = "gpt-4o-mini"
    
    memory_agent = ConversationMemoryAgent(
        storage_agent_model=storage_model,
        retrieval_agent_model=retrieval_model,
        orchestrator_agent_model=orchestrator_model
    )
    
    # Test 1: Memory Storage Performance
    logger.info("\n=== Test 1: Memory Storage Performance ===")
    test_messages = [
        "I love playing basketball",
        "My favorite color is blue",
        "I'm learning to play the guitar",
        "I have a dog named Max",
        "I work as a software engineer"
    ]
    
    for message in test_messages:
        start_time = time.time()
        try:
            await memory_agent.process_message(message, "user", "test")
            metrics.add_metric('memory_operations', time.time() - start_time)
        except Exception as e:
            logger.error(f"Error in memory storage: {e}")
            metrics.metrics['error_count'] += 1
    
    # Test 2: Memory Retrieval Performance
    logger.info("\n=== Test 2: Memory Retrieval Performance ===")
    test_queries = [
        "What sports do I like?",
        "What's my favorite color?",
        "What am I learning?",
        "Do I have any pets?",
        "What's my job?"
    ]
    
    for query in test_queries:
        start_time = time.time()
        try:
            await memory_agent.process_message(query, "user", "test")
            metrics.add_metric('response_times', time.time() - start_time)
        except Exception as e:
            logger.error(f"Error in memory retrieval: {e}")
            metrics.metrics['error_count'] += 1
    
    # Test 3: API Call Performance
    logger.info("\n=== Test 3: API Call Performance ===")
    api_operations = [
        "get_user_profile",
        "update_user_settings",
        "get_conversation_history",
        "save_conversation",
        "get_memory_stats"
    ]
    
    for operation in api_operations:
        start_time = time.time()
        try:
            # Simulate API call
            await asyncio.sleep(0.1)  # Simulate network delay
            metrics.add_metric('api_calls', time.time() - start_time)
        except Exception as e:
            logger.error(f"Error in API operation: {e}")
            metrics.metrics['error_count'] += 1
    
    # Print performance summary
    summary = metrics.get_summary()
    logger.info("\n=== Performance Summary ===")
    logger.info(f"Average Response Time: {summary['avg_response_time']:.3f} seconds")
    logger.info(f"Average Memory Operation Time: {summary['avg_memory_operation_time']:.3f} seconds")
    logger.info(f"Average API Call Time: {summary['avg_api_call_time']:.3f} seconds")
    logger.info(f"Total Errors: {summary['total_errors']}")
    
    return summary

if __name__ == "__main__":
    asyncio.run(test_memory_agent_performance()) 