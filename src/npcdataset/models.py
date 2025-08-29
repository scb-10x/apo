"""Core data models for NPC conversation datasets."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


@dataclass
class Message:
    """A single message in a conversation."""
    speaker: str
    text: str
    target_items: List[Dict[str, str]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create a Message from a dictionary."""
        return cls(
            speaker=data.get("speaker", ""),
            text=data.get("text", ""),
            target_items=data.get("target_item", [])
        )


@dataclass
class FunctionCall:
    """Represents a function call with parameters and return values."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    return_values: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionCall':
        """Create a FunctionCall from a dictionary."""
        return cls(
            name=data["name"],
            parameters=data.get("parameters", {}),
            return_values=data.get("return", [])
        )


@dataclass
class Turn:
    """
    Represents a single turn in the conversation with related data.
    
    Instead of storing message objects directly, this keeps track of
    message indices in the parent conversation's message stream.
    """
    # Indices into the parent conversation's message_stream
    message_indices: List[int]
    gold_response: str = ""
    gold_functions: List[FunctionCall] = field(default_factory=list)
    
    # Reference to parent conversation's message stream (set after creation)
    _message_stream: List[Message] = field(default_factory=list, repr=False)
    
    @property
    def messages(self) -> List[Message]:
        """Get the messages in this turn."""
        if not self._message_stream:
            return []
        return [self._message_stream[idx] for idx in self.message_indices]
    
    @property
    def last_message(self) -> Optional[Message]:
        """Get the last message in this turn."""
        if not self.message_indices or not self._message_stream:
            return None
        return self._message_stream[self.message_indices[-1]]


    
@dataclass
class Persona:
    """A character persona with various attributes."""
    name: str = ""
    age: str = ""
    gender: str = ""
    occupation: str = ""
    appearance: str = ""
    hobbies: str = ""
    personality_traits: str = ""
    background: str = ""
    special_skills: str = ""
    past_experiences: str = ""
    future_goals: str = ""
    daily_routines: str = ""
    
    # Additional attributes can be stored here
    additional_attributes: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Persona':
        """Create a Persona from a dictionary."""
        # Map known fields
        known_fields = {
            "name": data.get("name", ""),
            "age": data.get("age", ""),
            "gender": data.get("gender", ""),
            "occupation": data.get("occupation", ""),
            "appearance": data.get("appearance", ""),
            "hobbies": data.get("hobbies", ""),
            "personality_traits": data.get("personality traits", ""),
            "background": data.get("background", ""),
            "special_skills": data.get("strong points/special skills", ""),
            "past_experiences": data.get("past experiences", ""),
            "future_goals": data.get("goals and plans for the future", ""),
            "daily_routines": data.get("daily routines", ""),
        }
        
        # Store all additional fields
        additional = {}
        for k, v in data.items():
            if k not in [
                "name", "age", "gender", "occupation", "appearance", 
                "hobbies", "personality traits", "background", 
                "strong points/special skills", "past experiences",
                "goals and plans for the future", "daily routines"
            ]:
                additional[k] = v
                
        known_fields["additional_attributes"] = additional
        return cls(**known_fields)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert the Persona to a dictionary."""
        result = {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            "appearance": self.appearance,
            "hobbies": self.hobbies,
            "personality traits": self.personality_traits,
            "background": self.background,
            "strong points/special skills": self.special_skills,
            "past experiences": self.past_experiences,
            "goals and plans for the future": self.future_goals,
            "daily routines": self.daily_routines,
        }
        
        # Add any additional attributes
        result.update(self.additional_attributes)
        
        return result


@dataclass
class Conversation:
    """An iterable conversation with sequential turns and parallel gold data."""
    id: str
    message_stream: List[Message]
    turns: List[Turn]
    worldview: str = ""
    personas: Dict[str, Persona] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)
    knowledge: List[Dict[str, str]] = field(default_factory=list)
    general_knowledge: str = ""
    function_list_id: str = ""
    state: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set up the turns with references to the message stream."""
        for turn in self.turns:
            turn._message_stream = self.message_stream
    
    def __len__(self) -> int:
        """Get the number of turns in the conversation."""
        return len(self.turns)
    
    def __getitem__(self, idx) -> Union[Turn, List[Turn]]:
        """
        Get a specific turn or a slice of turns.
        This allows both conversation[i] and conversation[:i] syntax.
        """
        return self.turns[idx]
    
    def __iter__(self) -> Iterator[Turn]:
        """Make the conversation iterable over its turns."""
        return iter(self.turns)
    
    @property
    def all_messages(self) -> List[Message]:
        """Get all messages in the conversation."""
        return self.message_stream
    
    @property
    def gold_responses(self) -> List[str]:
        """Get all gold responses as a list."""
        return [turn.gold_response for turn in self.turns]
    
    @property
    def gold_functions(self) -> List[List[FunctionCall]]:
        """Get all gold function calls as a list of lists."""
        return [turn.gold_functions for turn in self.turns]
    
    def get_message_history(self, turn_idx: int, include_current: bool = False) -> List[Message]:
        """
        Get message history up to the specified turn.
        
        Args:
            turn_idx: The turn index to get history up to
            include_current: Whether to include messages from the current turn
            
        Returns:
            List of messages in chronological order
        """
        if turn_idx < 0 or turn_idx >= len(self.turns):
            raise IndexError(f"Turn index {turn_idx} out of range")
            
        if turn_idx == 0 and not include_current:
            return []
            
        # Find the index before the current turn
        prev_max_idx = -1
        if turn_idx > 0:
            prev_max_idx = max(self.turns[turn_idx-1].message_indices)
            
        if not include_current:
            return self.message_stream[:prev_max_idx + 1]
            
        # Include current turn messages
        current_max_idx = max(self.turns[turn_idx].message_indices) if self.turns[turn_idx].message_indices else -1
        return self.message_stream[:current_max_idx + 1]


@dataclass
class ConversationDataset:
    """A collection of conversations that can be accessed like a dictionary."""
    conversations: Dict[str, Conversation] = field(default_factory=dict)
    name: str = ""
    description: str = ""
    version: str = "1.0"
    
    def __getitem__(self, conversation_id: str) -> Conversation:
        """Get a conversation by ID using dictionary syntax."""
        if conversation_id not in self.conversations:
            raise KeyError(f"Conversation '{conversation_id}' not found")
        return self.conversations[conversation_id]
    
    def __iter__(self) -> Iterator[Conversation]:
        """Make the dataset iterable over conversations."""
        return iter(self.conversations.values())
    
    def __len__(self) -> int:
        """Get the number of conversations in the dataset."""
        return len(self.conversations)
    
    def add_conversation(self, conversation: Conversation) -> None:
        """Add a conversation to the dataset."""
        self.conversations[conversation.id] = conversation
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ConversationDataset':
        """Load conversations from a JSON file."""
        path = Path(json_path) if isinstance(json_path, str) else json_path
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Import here to avoid circular imports
        from npcdataset.parsers import parse_conversation_data
        return parse_conversation_data(data, name=path.stem)
    
    @classmethod
    def create(cls, name: str, description: str = "", version: str = "1.0") -> 'ConversationDataset':
        """Create a new empty dataset with metadata."""
        return cls(
            name=name,
            description=description,
            version=version
        )
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save the dataset to a JSON file."""
        path = Path(output_path) if isinstance(output_path, str) else output_path
        
        # Convert dataset to serializable dictionary
        data = []
        
        for conversation in self.conversations.values():
            conv_data = {
                "data_id": conversation.id,
                "total_turn": len(conversation.turns),
                "worldview": conversation.worldview,
                "player": {"persona": conversation.personas.get("player", Persona()).to_dict()},
                "npc": {"role": conversation.roles["npc"], "persona": conversation.personas.get("npc", Persona()).to_dict()},
                "function_list_id": conversation.function_list_id,
                "knowledge": {"knowledge_info": conversation.knowledge, "general_info": conversation.general_knowledge},
                "state": conversation.state,
            }
            
            # Add turn data
            for i, turn in enumerate(conversation.turns):
                turn_key = f"turn_{i}"
                
                turn_data = {
                    "dialogue": [
                        {
                            "speaker": msg.speaker,
                            "text": msg.text,
                            "target_item": msg.target_items
                        }
                        for msg in turn.messages
                    ],
                    "gold_response": turn.gold_response,
                    "gold_functions": [
                        {
                            "name": func.name,
                            "parameters": func.parameters,
                            "return": func.return_values
                        }
                        for func in turn.gold_functions
                    ]
                }
                
                conv_data[turn_key] = turn_data
            
            data.append(conv_data)
        
        # Save to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def filter(self, predicate) -> 'ConversationDataset':
        """Filter conversations based on a predicate function."""
        result = self.__class__(name=f"{self.name}_filtered", version=self.version)
        for conv_id, conv in self.conversations.items():
            if predicate(conv):
                result.add_conversation(conv)
        return result
