"""Utility functions for working with NPC datasets."""

from typing import List, Dict, Any, Optional
import inspect
from pathlib import Path
import importlib.util


def load_tools_from_module(module_path: str) -> Dict[str, Any]:
    """
    Dynamically load all tools and actions from a Python module.
    
    Args:
        module_path: Path to the Python module file
        
    Returns:
        Dictionary containing tool and action functions
    """
    # Import the tools module
    path = Path(module_path)
    module_name = path.stem
    
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    tools = {}
    actions = {}
    
    # Extract tools and actions
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            if hasattr(obj, '_is_tool') and obj._is_tool:
                tools[name] = {
                    "function": obj,
                    "name": name,
                    "description": getattr(obj, '_tool_description', obj.__doc__ or ""),
                    "parameters": _extract_function_parameters(obj)
                }
            elif hasattr(obj, '_is_action') and obj._is_action:
                actions[name] = {
                    "function": obj,
                    "name": name,
                    "description": getattr(obj, '_action_description', obj.__doc__ or ""),
                    "parameters": _extract_function_parameters(obj)
                }
    
    return {"tools": tools, "actions": actions}


def _extract_function_parameters(func) -> Dict[str, Dict[str, Any]]:
    """Extract parameter information from a function's signature and docstring."""
    sig = inspect.signature(func)
    params = {}
    
    for name, param in sig.parameters.items():
        # Skip knowledge_base parameter as it's implicit
        if name == 'knowledge_base':
            continue
            
        param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "string"
        param_default = None if param.default == inspect.Parameter.empty else param.default
        
        # Extract description from docstring if available
        desc = _extract_param_doc(func, name)
        
        params[name] = {
            "type": param_type,
            "description": desc,
            "default": param_default
        }
    
    return params


def _extract_param_doc(func, param_name: str) -> str:
    """Extract parameter documentation from function docstring."""
    if not func.__doc__:
        return ""
        
    lines = func.__doc__.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(param_name + ':') or line.startswith(param_name + ' :'):
            return line.split(':', 1)[1].strip()
    return ""


def format_for_training(dataset, format: str = "openai") -> List[Dict[str, Any]]:
    """Format the dataset for training in various ML frameworks."""
    # Implementation would depend on the specific format needed
    pass
