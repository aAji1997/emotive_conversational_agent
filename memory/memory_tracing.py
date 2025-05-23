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

def initialize_tracing(project_name: str,
                       run_name: Optional[str] = None,
                       enabled: bool = True) -> bool:
    """
    Initialize the tracing system.

    Args:
        project_name: W&B project name (required)
        run_name: Optional name for this tracing run
        enabled: Whether tracing is enabled

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if not project_name:
        logger.error("Project name is required for tracing initialization")
        return False

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
        weave.init(project_name=project_name)

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

            # Create a name for the operation
            op_name = f"{component_name}.{method_type}"

            # Add attributes
            attributes = {
                "component": component_name,
                "method_type": method_type
            }

            # Add arguments (safely)
            safe_args = _sanitize_args(args, kwargs)

            # Use the weave.op decorator to create a traced function
            @weave.op(name=op_name)
            def traced_func():
                # Call the original function
                return func(*args, **kwargs)

            # Call the traced function with attributes
            with weave.attributes(attributes):
                result = traced_func()

            return result

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
        # Create a name for the operation
        op_name = f"{source_component}_to_{target_component}"

        # Add attributes
        attributes = {
            "source": source_component,
            "target": target_component,
            "interaction_type": interaction_type,
            "timestamp": time.time()
        }

        # Prepare data
        safe_data = _sanitize_result(data) if data else {}

        # Use the weave.op decorator to create a traced function
        @weave.op(name=op_name)
        def traced_interaction():
            return {}

        # Call the traced function with attributes
        with weave.attributes(attributes):
            traced_interaction()
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
