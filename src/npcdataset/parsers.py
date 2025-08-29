"""Parsers for loading conversation data from different formats."""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_conversation_data(data: Union[List[Dict[str, Any]], Dict[str, Any]], name: str = "") -> 'ConversationDataset':
    """
    Parse conversation data from a dictionary or list of dictionaries.
    
    Args:
        data: Dictionary or list of dictionaries containing conversation data
        name: Name for the dataset
        
    Returns:
        ConversationDataset containing the parsed conversations
    """
    # Import here to avoid circular imports
    from npcdataset.models import (
        Conversation,
        ConversationDataset,
        FunctionCall,
        Message,
        Persona,
        Turn,
    )

    # Create an empty dataset
    dataset = ConversationDataset(name=name)
    
    # Handle both single conversation and list of conversations
    conversations_data = data if isinstance(data, list) else [data]
    
    for conv_data in conversations_data:
        # Extract conversation ID
        conv_id = conv_data.get("data_id", f"conversation_{len(dataset.conversations)}")

        # Extract shared data
        worldview = conv_data.get("worldview", "")

        # Extract personas
        personas = {}
        if "player" in conv_data and "persona" in conv_data["player"]:
            personas["player"] = Persona.from_dict(conv_data["player"]["persona"])
        if "npc" in conv_data and "persona" in conv_data["npc"]:
            personas["npc"] = Persona.from_dict(conv_data["npc"]["persona"])

        # Extract role
        roles = {}
        if "player" in conv_data and "role" in conv_data["player"]:
            roles["player"] = conv_data["player"]["role"]
        if "npc" in conv_data and "role" in conv_data["npc"]:
            roles["npc"] = conv_data["npc"]["role"]
            
        # Extract knowledge
        knowledge = []
        general_knowledge = ""
        if "knowledge" in conv_data:
            if "knowledge_info" in conv_data["knowledge"]:
                knowledge = conv_data["knowledge"]["knowledge_info"]
            if "general_info" in conv_data["knowledge"]:
                general_knowledge = conv_data["knowledge"]["general_info"]
            
        # Extract function information
        function_list_id = conv_data.get("function_list_id", "")

        # Extract state
        state = {}
        if "state" in conv_data:
            state = conv_data["state"]
        
        # Extract turn keys and sort them
        turn_keys = sorted(
            [k for k in conv_data.keys() if k.startswith("turn_")],
            key=lambda k: int(k.split("_")[1])
        )
        
        if not turn_keys:
            continue            

        
#        verify_data_consistency(conv_data, turn_keys, first_turn_data)
        
        # Create a single message stream for the conversation
        message_stream = []
        turns = []
        
        for i, turn_key in enumerate(turn_keys):
            turn_data = conv_data[turn_key]
            
            # Create messages for this turn
            turn_messages = []
            if "dialogue" in turn_data:
                for msg_data in turn_data["dialogue"]:
                    turn_messages.append(Message.from_dict(msg_data))
            
            # Start index for this turn's messages in the stream
            message_offset = len(message_stream)
            
            # Generate message indices
            message_indices = list(range(message_offset, message_offset + len(turn_messages)))
            
            # Parse gold functions
            gold_functions = []
            if "gold_functions" in turn_data:
                for func_data in turn_data["gold_functions"]:
                    # Create proper FunctionCall objects
                    gold_functions.append(FunctionCall.from_dict(func_data))
            
            # Add messages to stream
            message_stream.extend(turn_messages)
            
            # Create turn
            turn = Turn(
                message_indices=message_indices,
                gold_response=turn_data.get("gold_response", ""),
                gold_functions=gold_functions,  # Store gold functions
            )
            
            turns.append(turn)
        
        # Create conversation
        conversation = Conversation(
            id=conv_id,
            message_stream=message_stream,
            turns=turns,
            worldview=worldview,
            personas=personas,
            roles=roles,
            knowledge=knowledge,
            general_knowledge=general_knowledge,
            function_list_id=function_list_id,
            state=state
        )
        
        # Add to dataset
        dataset.add_conversation(conversation)
    
    return dataset


def verify_data_consistency(conv_data: Dict[str, Any], turn_keys: List[str], first_turn_data: Dict[str, Any]) -> None:
    """
    Verify that shared data is consistent across all turns.
    
    Args:
        conv_data: Conversation data dictionary
        turn_keys: List of turn key strings (e.g., "turn_0", "turn_1")
        first_turn_data: Data from the first turn for comparison
        
    Raises:
        Warning if inconsistencies are found
    """
    # Fields to check for consistency
    check_fields = [
        ("worldview", None),
        ("person_A", "persona"),
        ("person_B", "persona"),
        ("tool_functions", None),
        ("action_functions", None),
        ("knowledge", "knowledge_info")
    ]
    
    # Check all turns after the first
    for turn_key in turn_keys[1:]:
        turn_data = conv_data[turn_key]
        
        for field, subfield in check_fields:
            if field not in first_turn_data or field not in turn_data:
                continue
                
            first_value = first_turn_data[field]
            current_value = turn_data[field]
            
            if subfield:
                if subfield in first_value and subfield in current_value:
                    first_value = first_value[subfield]
                    current_value = current_value[subfield]
                else:
                    continue
            
            # Simple equality check (this could be enhanced for more complex comparisons)
            if first_value != current_value:
                warnings.warn(
                    f"Inconsistency found in {turn_key} for field {field}"
                    f"{f'.{subfield}' if subfield else ''}"
                )
