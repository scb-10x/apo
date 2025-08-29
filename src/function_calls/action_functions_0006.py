from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function


@tool 
def select_request_record(quest_name: str) -> None:
    """
    Record the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.) as a potential selection.

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def select(quest_name: str) -> None:
    """
    Select the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.) or quests recorded as potential selection.

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

@tool 
def start(quest_name: str) -> None:
    """
    Start the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    None
    """

    pass

all_functions = [select_request_record, select, start]
action_functions_0006 = {'function_registry': {
    f.name: convert_to_openai_function(f, strict=True) for f in all_functions
}}