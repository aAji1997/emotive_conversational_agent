"""
Memory Tracing Module for visualizing memory component interactions.

This module provides tracing functionality using Weights & Biases (W&B) Weave
to visualize the interactions between memory storage, retrieval, and orchestrator
components in the conversational agent.
"""

import os
import time
import logging
import functools
import json
from typing import Dict, Any, Optional, Callable, List

# Import Weave for tracing
try:
    import wandb
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    print("WARNING: wandb or weave not installed. Tracing functionality will be disabled.")
    print("To enable tracing, install with: pip install wandb weave")

# Set up logging
logger = logging.getLogger(__name__)

# Global trace context
trace_context = {
    "initialized": False,
    "project_name": "memory-agent-tracing",
    "run_id": None,
    "traces": [],
    "enabled": False
}

def initialize_tracing(project_name: str = "memory-agent-tracing", 
                       run_name: Optional[str] = None,
                       enabled: bool = True) -> bool:
    """
    Initialize the tracing system.
    
    Args:
        project_name: W&B project name
        run_name: Optional name for this tracing run
        enabled: Whether tracing is enabled
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if not WEAVE_AVAILABLE:
        logger.warning("Weave not available. Tracing disabled.")
        return False
    
    trace_context["enabled"] = enabled
    if not enabled:
        logger.info("Tracing is disabled by configuration.")
        return False
    
    try:
        # Initialize W&B
        if run_name is None:
            run_name = f"memory-trace-{int(time.time())}"
            
        trace_context["project_name"] = project_name
        trace_context["run_id"] = run_name
        
        # Initialize wandb
        wandb.init(project=project_name, name=run_name)
        
        # Initialize weave tracing
        weave.init()
        
        trace_context["initialized"] = True
        logger.info(f"Tracing initialized for project '{project_name}', run '{run_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        return False

def end_tracing():
    """End the current tracing session."""
    if not trace_context["initialized"] or not WEAVE_AVAILABLE:
        return
    
    try:
        # Finish the wandb run
        wandb.finish()
        trace_context["initialized"] = False
        logger.info("Tracing session ended")
    except Exception as e:
        logger.error(f"Error ending tracing session: {e}")

def trace_method(component_name: str, method_type: str):
    """
    Decorator for tracing method calls in memory components.
    
    Args:
        component_name: Name of the component (storage, retrieval, orchestrator)
        method_type: Type of method (e.g., 'store', 'retrieve', 'decide')
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not trace_context["initialized"] or not trace_context["enabled"] or not WEAVE_AVAILABLE:
                return func(*args, **kwargs)
            
            # Start span for this method
            with weave.trace(f"{component_name}.{method_type}") as span:
                # Add attributes to the span
                span.set_attribute("component", component_name)
                span.set_attribute("method_type", method_type)
                
                # Add arguments to the span (safely)
                safe_args = _sanitize_args(args, kwargs)
                span.set_attribute("args", json.dumps(safe_args))
                
                # Record the start time
                start_time = time.time()
                
                try:
                    # Call the original function
                    result = func(*args, **kwargs)
                    
                    # Record the result (safely)
                    safe_result = _sanitize_result(result)
                    span.set_attribute("result", json.dumps(safe_result))
                    span.set_attribute("status", "success")
                    
                    return result
                except Exception as e:
                    # Record the exception
                    span.set_attribute("error", str(e))
                    span.set_attribute("status", "error")
                    raise
                finally:
                    # Record the duration
                    span.set_attribute("duration_ms", (time.time() - start_time) * 1000)
        
        return wrapper
    
    return decorator

def trace_interaction(source_component: str, target_component: str, interaction_type: str, 
                     data: Optional[Dict[str, Any]] = None):
    """
    Trace an interaction between two components.
    
    Args:
        source_component: Name of the source component
        target_component: Name of the target component
        interaction_type: Type of interaction (e.g., 'request', 'response')
        data: Optional data associated with the interaction
    """
    if not trace_context["initialized"] or not trace_context["enabled"] or not WEAVE_AVAILABLE:
        return
    
    try:
        with weave.trace(f"{source_component}_to_{target_component}") as span:
            span.set_attribute("source", source_component)
            span.set_attribute("target", target_component)
            span.set_attribute("interaction_type", interaction_type)
            
            if data:
                safe_data = _sanitize_result(data)
                span.set_attribute("data", json.dumps(safe_data))
            
            span.set_attribute("timestamp", time.time())
    except Exception as e:
        logger.error(f"Error tracing interaction: {e}")

def _sanitize_args(args, kwargs):
    """Sanitize arguments for safe logging."""
    safe_args = {}
    
    # Process positional arguments
    for i, arg in enumerate(args):
        if i == 0 and arg.__class__.__name__ == 'ConversationMemoryAgent':
            # Skip self argument
            continue
        safe_args[f"arg_{i}"] = _sanitize_value(arg)
    
    # Process keyword arguments
    for key, value in kwargs.items():
        safe_args[key] = _sanitize_value(value)
    
    return safe_args

def _sanitize_result(result):
    """Sanitize result for safe logging."""
    return _sanitize_value(result)

def _sanitize_value(value):
    """Sanitize a value for safe logging."""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    else:
        # For complex objects, just return the class name and string representation
        return f"{value.__class__.__name__}: {str(value)[:100]}"

# Visualization helper functions
def create_trace_visualization(output_file: str = "memory_trace_visualization.html"):
    """
    Create a visualization of the traced interactions.
    
    Args:
        output_file: Path to save the HTML visualization
    
    Returns:
        str: Path to the generated visualization file
    """
    if not WEAVE_AVAILABLE or not trace_context["initialized"]:
        logger.warning("Weave not available or tracing not initialized. Cannot create visualization.")
        return None
    
    try:
        # Get the trace data from Weave
        traces = weave.get_traces()
        
        # Generate the visualization
        html_content = weave.generate_trace_visualization(traces)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Trace visualization saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error creating trace visualization: {e}")
        return None
