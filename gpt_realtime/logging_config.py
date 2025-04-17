"""
Logging configuration for the GPT Realtime application.
This module configures the logging levels for different components of the application.
"""

import logging

def configure_logging():
    """Configure logging for the application."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers to Error level to reduce verbosity
    logging.getLogger('memory.memory_agent').setLevel(logging.ERROR)
    logging.getLogger('memory.memory_integration').setLevel(logging.ERROR)
    logging.getLogger('LiteLLM').setLevel(logging.ERROR)
    logging.getLogger('websockets').setLevel(logging.ERROR)
    
    # Suppress model and API related logging
    logging.getLogger('openai').setLevel(logging.ERROR)
    logging.getLogger('google.generativeai').setLevel(logging.ERROR)
    logging.getLogger('google.adk').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    
    # Suppress the realtime client's verbose logging
    logging.getLogger('gpt_realtime.realtime_audio_gpt').setLevel(logging.ERROR)
    logging.getLogger('gpt_realtime').setLevel(logging.ERROR)
    
    # Suppress Dash debug logs
    logging.getLogger('dash').setLevel(logging.ERROR)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)


