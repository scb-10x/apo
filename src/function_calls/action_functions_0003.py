from langchain.tools import tool
from typing import List, Dict, Tuple
from langchain_core.utils.function_calling import convert_to_openai_function


@tool 
def sell_request_record(item_name: str) -> None:
    """
    Record the specified weapon (e.g. Avis Wind, Short Sword, etc.) as a potential purchase.

    Parameters:
    ----------
    item_name: str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def sell(item_names: List[str]) -> None:
    """
    Sell the specified weapon (e.g. Avis Wind, Short Sword, etc.) or weapons recorded as potential purchases.

    Parameters:
    ----------
    item_name: List[str]
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def equip(item_name: str) -> None:
    """
    Equip the specified weapon (e.g. Avis Wind, Short Sword, etc.).

    Parameters:
    ----------
    item_name: str
        Specified weapon name (e.g. Avis Wind, Short Sword, etc.). Uses the weapon name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

all_functions = [sell_request_record, sell, equip]
action_functions_0003 = {'function_registry': {
    f.name: convert_to_openai_function(f, strict=True) for f in all_functions
}}