"""Tools and function handling for the NPC dataset."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolParameter:
    """Definition of a parameter for a tool."""
    name: str
    description: str
    type: str = "string"
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """Definition of a tool that can be used in conversations."""
    name: str
    description: str
    parameters: Dict[str, ToolParameter] = field(default_factory=dict)
    function: Optional[Callable] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Create a Tool from a dictionary."""
        params = {}
        if "parameters" in data and "properties" in data["parameters"]:
            for param_name, param_data in data["parameters"]["properties"].items():
                params[param_name] = ToolParameter(
                    name=param_name,
                    description=param_data.get("description", ""),
                    type=param_data.get("type", "string"),
                    required=param_name in data["parameters"].get("required", []),
                    enum=param_data.get("enum")
                )
                
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool to a dictionary."""
        params_dict = {
            "type": "object",
            "properties": {}
        }
        
        required_params = []
        
        for param_name, param in self.parameters.items():
            param_dict = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum:
                param_dict["enum"] = param.enum
                
            params_dict["properties"][param_name] = param_dict
            
            if param.required:
                required_params.append(param_name)
                
        if required_params:
            params_dict["required"] = required_params
            
        return {
            "name": self.name,
            "description": self.description,
            "parameters": params_dict
        }


@dataclass
class ToolRegistry:
    """Registry of tools and actions available for conversations."""
    tools: Dict[str, Tool] = field(default_factory=dict)
    actions: Dict[str, Tool] = field(default_factory=dict)
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        
    def register_action(self, action: Tool) -> None:
        """Register an action in the registry."""
        self.actions[action.name] = action
        
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
        
    def get_action(self, name: str) -> Optional[Tool]:
        """Get an action by name."""
        return self.actions.get(name)
    
    @classmethod
    def from_dicts(cls, tools_dicts: List[Dict[str, Any]], actions_dicts: List[Dict[str, Any]]) -> 'ToolRegistry':
        """Create a registry from dictionaries of tools and actions."""
        registry = cls()
        
        for tool_dict in tools_dicts:
            tool = Tool.from_dict(tool_dict)
            registry.register_tool(tool)
            
        for action_dict in actions_dicts:
            action = Tool.from_dict(action_dict)
            registry.register_action(action)
            
        return registry
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert tools and actions to OpenAI function calling format."""
        result = []
        
        for tool in self.tools.values():
            result.append({
                "type": "function",
                "function": tool.to_dict()
            })
            
        for action in self.actions.values():
            result.append({
                "type": "function",
                "function": action.to_dict()
            })
            
        return result


def tool(description: str = None):
    """Decorator to mark a function as an available tool."""
    def decorator(func):
        func._is_tool = True
        func._tool_description = description or func.__doc__
        return func
    return decorator


def action(description: str = None):
    """Decorator to mark a function as an available action."""
    def decorator(func):
        func._is_action = True
        func._action_description = description or func.__doc__
        return func
    return decorator


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
