import asyncio
import time
import logging
import random
from datetime import datetime
import sys
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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

class LoadTestMetrics:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'errors': [],
            'concurrent_users': 0,
            'total_requests': 0,
            'successful_requests': 0
        }
    
    def add_response_time(self, time_taken):
        self.metrics['response_times'].append(time_taken)
    
    def add_error(self, error):
        self.metrics['errors'].append(error)
    
    def increment_requests(self):
        self.metrics['total_requests'] += 1
    
    def increment_success(self):
        self.metrics['successful_requests'] += 1
    
    def get_summary(self):
        if not self.metrics['response_times']:
            return {
                'avg_response_time': 0,
                'error_rate': 1.0,
                'success_rate': 0.0,
                'total_requests': self.metrics['total_requests']
            }
        
        return {
            'avg_response_time': sum(self.metrics['response_times']) / len(self.metrics['response_times']),
            'error_rate': len(self.metrics['errors']) / self.metrics['total_requests'],
            'success_rate': self.metrics['successful_requests'] / self.metrics['total_requests'],
            'total_requests': self.metrics['total_requests']
        }

async def simulate_user_session(user_id, metrics, openai_api_key):
    """Simulate a single user session with multiple operations."""
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
    
    # Simulate user interactions
    messages = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a joke",
        "What's my name?",
        "Goodbye"
    ]
    
    for message in messages:
        metrics.increment_requests()
        start_time = time.time()
        try:
            await memory_agent.process_message(message, f"user_{user_id}", "test")
            metrics.add_response_time(time.time() - start_time)
            metrics.increment_success()
        except Exception as e:
            metrics.add_error(str(e))
            logger.error(f"Error in user session {user_id}: {e}")
        
        # Add random delay between requests
        await asyncio.sleep(random.uniform(0.1, 0.5))

async def run_load_test(num_users, duration_seconds):
    """Run a load test with specified number of users and duration."""
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
    
    metrics = LoadTestMetrics()
    start_time = time.time()
    
    # Create tasks for each user
    tasks = []
    for i in range(num_users):
        task = asyncio.create_task(simulate_user_session(i, metrics, openai_api_key))
        tasks.append(task)
    
    # Wait for all tasks to complete or until duration is reached
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration_seconds)
    except asyncio.TimeoutError:
        logger.info("Load test duration reached")
    
    # Print test summary
    summary = metrics.get_summary()
    logger.info("\n=== Load Test Summary ===")
    logger.info(f"Number of Users: {num_users}")
    logger.info(f"Test Duration: {duration_seconds} seconds")
    logger.info(f"Total Requests: {summary['total_requests']}")
    logger.info(f"Average Response Time: {summary['avg_response_time']:.3f} seconds")
    logger.info(f"Success Rate: {summary['success_rate']*100:.2f}%")
    logger.info(f"Error Rate: {summary['error_rate']*100:.2f}%")
    
    return summary

if __name__ == "__main__":
    # Run load test with 10 concurrent users for 60 seconds
    asyncio.run(run_load_test(num_users=10, duration_seconds=60)) 