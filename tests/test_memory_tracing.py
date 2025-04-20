"""
Test script for memory tracing and visualization.

This script demonstrates how to use the traced memory agent and visualization
to trace and visualize the interactions between memory components.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService

# Import the traced memory agent
from memory.memory_agent_traced import TracedConversationMemoryAgent
from memory.memory_tracing import initialize_tracing, end_tracing, create_trace_visualization

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_traced_memory_agent():
    """Test the traced memory agent."""
    try:
        # Load API key for OpenAI
        with open('.api_key.json', 'r') as f:
            api_keys = json.load(f)
            openai_api_key = api_keys.get('openai_api_key')
            supabase_url = api_keys.get('supabase_url')
            supabase_key = api_keys.get('supabase_api_key')

        if not openai_api_key:
            logger.error("OpenAI API key not found in .api_key.json")
            return False

        # Initialize models
        storage_model = LiteLlm(
            model="gpt-4o",
            api_key=openai_api_key,
            temperature=0.2
        )

        retrieval_model = LiteLlm(
            model="gpt-4o",
            api_key=openai_api_key,
            temperature=0.3
        )

        orchestrator_model = LiteLlm(
            model="gpt-4o",
            api_key=openai_api_key,
            temperature=0.1
        )

        # Initialize session service
        session_service = InMemorySessionService()

        # Initialize the traced memory agent
        memory_agent = TracedConversationMemoryAgent(
            storage_agent_model=storage_model,
            retrieval_agent_model=retrieval_model,
            orchestrator_agent_model=orchestrator_model,
            session_service=session_service,
            user_id="test_user",
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            enable_tracing=True,
            trace_project_name="memory-tracing-test"
        )

        logger.info("Traced memory agent initialized")

        # Process a few test messages
        test_messages = [
            "My name is John and I live in New York.",
            "I have a dog named Max.",
            "What's my dog's name?",
            "Where do I live?",
            "I'm allergic to peanuts."
        ]

        for message in test_messages:
            logger.info(f"Processing message: {message}")
            response, memories = await memory_agent.process_message(message)
            logger.info(f"Response: {response}")
            if memories:
                logger.info(f"Retrieved {len(memories)} memories")
                for i, memory in enumerate(memories):
                    logger.info(f"  Memory {i+1}: {memory.get('content')}")

        # Generate a visualization
        visualization_file = memory_agent.generate_visualization("memory_trace_visualization.html")
        if visualization_file:
            logger.info(f"Visualization generated: {visualization_file}")

        # Close the agent (and end tracing)
        memory_agent.close()

        return True
    except Exception as e:
        logger.error(f"Error testing traced memory agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def run_visualization_dashboard():
    """Run the memory visualization dashboard."""
    try:
        from memory.memory_visualization import MemoryVisualizationDashboard
        
        # Create and run the dashboard
        dashboard = MemoryVisualizationDashboard()
        dashboard.run_server(debug=True, port=8050)
        
        return True
    except Exception as e:
        logger.error(f"Error running visualization dashboard: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main entry point for the test script."""
    # Test the traced memory agent
    success = await test_traced_memory_agent()
    
    if success:
        logger.info("Traced memory agent test completed successfully")
        
        # Run the visualization dashboard
        logger.info("Starting visualization dashboard...")
        await run_visualization_dashboard()
    else:
        logger.error("Traced memory agent test failed")

if __name__ == "__main__":
    asyncio.run(main())
