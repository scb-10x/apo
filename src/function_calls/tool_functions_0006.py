from typing import List, Dict
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function


@tool 
def search_quest(quest_name: str, quest_level: str, quest_duration: str, quest_reward: str, quest_description: str,
                 quest_name_operator: str, quest_level_operator: str, quest_duration_operator: str, quest_reward_operator: str) -> List[Dict[str, str]]:
    """
    Search for quests based on specified criteria, such as level(e.g. A, B,  etc.), duration (e.g. 2 hours, 3 days, etc.), reward (e.g. 2G, 10G, etc.), and specific features (e.g. investigation-type, can test one's magical abilities, etc.). Returns a list of quest names along with the reasons for the selection. Returns 'many' when there are multiple applicable items, and 'n/a' when there are none.

    Parameters:
    ----------
    quest_name: str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation. Multiple quests can be set (e.g.  Collecting Medical Herbs|Collecting Dragon Teardrops).

    quest_level: str
        Specified quest level (e.g. A, B, etc.). Uses the level of the quest mentioned in the conversation. Multiple levels can be set (e.g. A|B).

    quest_duration: str
        Specified quest duration (e.g. 2 hours, 3 Days, etc.). Uses the duration(days) of the quest mentioned in the conversation.

    quest_reward: str
        Specified quest reward (e.g. 10G, 500G, etc.). Uses the reward of the quest mentioned in the conversation.

    quest_description: str
        Specified quest characteristics (e.g. investigation-type, can test one's magica abilities, etc.). Uses the characteristics of the quest mentioned in the conversation.

    quest_name_operator: str
        Exclusion modifier used with the quest name specified by quest_name. Uses 'other than' as the modifier.

    quest_level_operator: str
        Modifier for comparison and exclusion used to describe the level of the quest specified by quest_level. The modifier can be one of the following: or above, or below, more than, less than, most difficult, difficult, average, easy, sesiest, other than.

    quest_duration_operator: str
        Modifier for comparison used to describe the duration of the quest specified by quest_duration. The modifier can be one of the following: or more, or less, more than, less than, about, longest, long, average, short, shortest.

    quest_reward_operator: str
        Modifier for comparison to describe the reward of the quest specified by quest_reward. The modifier can be one of the following: or more, or less, more than, less than, about, highest, high, average, low, lowest

    Returns:
    -------
    List[Dict[str, str]]
        A list of quest names along with the reasons for the selection.        

    """

    pass



@tool 
def check_basic_info(quest_name: str) -> List[Dict[str, str]]:
    """
    Check the level, duration, reward, and basic information of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name : str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the level, duration, reward, and basic information of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    """

    pass

@tool 
def check_level(quest_name: str) -> List[Dict[str, str]]:
    """
    Check the level of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name : str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the level of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    """

    pass

@tool 
def check_duration(quest_name: str) -> List[Dict[str, str]]:
    """
    Check the duration (hours) of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    
    Parameters:
    ----------
    quest_name : str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the duration (hours) of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    """

    pass


@tool 
def check_reward(quest_name: str) -> List[Dict[str, str]]:
    """
    Check the reward of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name : str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs the reward of a specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    """

    pass

@tool 
def check_description(quest_name: str) -> List[Dict[str, str]]:
    """
    Check the basic information and additional detailed information of the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).

    Parameters:
    ----------
    quest_name : str
        Specified quest name (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops, etc.). Uses the quest name mentioned in the conversation.

    Returns:
    -------
    List[Dict[str, str]]
        Outputs  the basic information and additional detailed information of the specified quest (e.g. Collecting Medical Herbs, Collecting Dragon Teardrops , etc.).
    """

    pass

all_functions = [search_quest, check_basic_info, check_level, check_duration, check_reward, check_description]
tool_functions_0006 = {'function_registry': {
    f.name: convert_to_openai_function(f, strict=True) for f in all_functions
}}