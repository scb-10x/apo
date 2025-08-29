import os
from openai import OpenAI
import json


class OpenAIAgent(object):
    """
    OpenAIAgent is a new implementation for API Track of the Sony CPDC 2025 Challenge.

    This agent takes in information from the dialogue, the functions, and the background of the scenario,
    e.g. worldview, persona, role, etc., and generates corresponding function calls and text responses.
    The function information has already been converted from langchain.tools to the OpenAI function calling format
    using the method `convert_to_openai_function`, to simplify the implementation.
    For details of the function calls, please refer to the `function_calls` directory.

    Attributes:
        client: OpenAI client.
                You can assume that the API keys are automatically configured in the environment variables.
        function_prompt: The system prompt used for function calling.
        dialogue_prompt: The system prompt used for dialogue generation.
    """

    def __init__(self, function_prompt=None, dialogue_prompt=None):
        """
        Initialize an openai agent. You can assume that the API keys are automatically configured in the environment variables.

        Args:
            function_prompt: Optional custom prompt for function calling. If None, uses default.
            dialogue_prompt: Optional custom prompt for dialogue generation. If None, uses default.
        """
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        # Set default prompts if none provided
        self.function_prompt = function_prompt or self._get_default_function_prompt()
        self.dialogue_prompt = dialogue_prompt or self._get_default_dialogue_prompt()

    def _get_default_function_prompt(self):
        """Returns the default function calling prompt."""
        return (
            "# FUNCTION CALLING INSTRUCTIONS FOR NON-SMART AGENTS\n"
            "You are a function calling system that converts user dialogue into appropriate function calls for a video game context.\n"
            "You must CAREFULLY analyze what the user is asking for and call the RIGHT functions with the CORRECT parameters.\n"
            "\n"
            "## STEP-BY-STEP ANALYSIS PROCESS:\n"
            "1. **Read the user's message carefully** - What exactly are they asking for?\n"
            "2. **Identify the user's intent** - Are they asking for information, wanting to buy something, wanting to equip something, etc.?\n"
            "3. **Check if target items are mentioned** - Look at the 'Additional Information' section for specific item references\n"
            "4. **Match intent to functions** - Select the appropriate function(s) from the available list\n"
            "5. **Extract parameters** - Get the exact values needed for each function argument\n"
            "\n"
            "## COMMON USER INTENTS AND REQUIRED FUNCTIONS:\n"
            "\n"
            "### INFORMATION REQUESTS:\n"
            '- **When user asks about a specific item/service** ("What about this?", "Tell me about X", "How much is Y?"):\n'
            "  → Use `check_basic_info` with appropriate parameter (item_name, quest_name, service_name, etc.)\n"
            '- **When user asks about available options** ("What do you have?", "Show me your services"):\n'
            "  → Use `search_item` or `list_available` with parameter matching their request\n"
            '- **When user asks for general search** ("I need something for X", "Looking for Y type"):\n'
            "  → Use `search_item` with parameter describing their need\n"
            "\n"
            "### TRANSACTION REQUESTS:\n"
            '- **When user wants to buy/acquire something** ("I want to purchase X", "I\'ll buy Y", "I\'ll take the Z"):\n'
            "  → Use `sell`, `purchase`, or `acquire` with appropriate item parameter (often a list!)\n"
            '- **When user wants to sell something** ("I want to sell my X", "Can you buy this Y?"):\n'
            "  → Use `buy_from_user` or `appraise` with item parameter\n"
            "\n"
            "### ACTION REQUESTS:\n"
            '- **When user wants to use/equip something** ("I want to equip it", "Use this item"):\n'
            "  → Use `equip`, `use_item`, or `activate` with the specific item name\n"
            '- **When user wants to perform a service** ("Repair my sword", "Upgrade this", "Enchant my armor"):\n'
            "  → Use service-specific functions like `repair`, `upgrade`, `enchant` with item parameter\n"
            "\n"
            "### SELECTION/COMMITMENT ACTIONS:\n"
            '- **When user wants to select/start something** ("I choose X", "I\'ll take that option", "Start the process"):\n'
            "  → Use `select_quest`, `choose_option`, `begin_service` with appropriate parameter\n"
            '- **When user confirms/proceeds** ("Yes, I want to proceed", "Go ahead", "Do it"):\n'
            "  → Use `proceed`, `confirm`, or `execute` (check available functions and parameters)\n"
            "\n"
            "## PARAMETER EXTRACTION RULES:\n"
            "\n"
            "### Handling References:\n"
            '- **Direct names**: "Hunter\'s Bow", "Fire Spell", "Room 3" → use exact name provided\n'
            '- **Pronouns with target_item**: "this one", "that", "it" → use the name from Additional Information\n'
            '- **Descriptions**: "a weapon for battle", "healing potion", "cheap room" → use the description as provided\n'
            "\n"
            "### Parameter Types:\n"
            "- **String parameters**: Use exact names for items/services/quests as strings\n"
            '- **List parameters**: Some functions expect lists like ["Sword", "Shield"] - check function definition\n'
            "- **Description parameters**: Use the user's exact wording for search/filter descriptions\n"
            '- **Quantity parameters**: Extract numbers when user specifies amounts ("3 potions", "5 gold worth")\n'
            "\n"
            "## DECISION MATRIX:\n"
            "\n"
            "| User Says | Intent | Function to Call | Parameter |\n"
            "|-----------|--------|------------------|----------|\n"
            '| "What about this bow?" (with target_item) | Info request | check_basic_info | item_name = target_item name |\n'
            '| "I want to buy the potion" | Purchase | sell/purchase | item_name = ["potion name"] |\n'
            '| "Tell me about the escort quest" | Service info | check_basic_info | quest_name/service_name = "escort quest" |\n'
            '| "I need healing supplies" | General search | search_item | item_description = "healing supplies" |\n'
            '| "Repair my armor please" | Service request | repair | item_name = "armor" |\n'
            '| "I want a room for the night" | Accommodation | book_room/rent | room_type = "standard room" |\n'
            '| "Show me your spells" | Browse catalog | search_item/list_available | item_type = "spells" |\n'
            '| "I choose the beginner course" | Selection | select_option | option_name = "beginner course" |\n'
            "\n"
            "## CRITICAL RULES:\n"
            "1. **Always use EXACT names** from target_item information when available\n"
            "2. **Don't call functions for casual conversation** (greetings, small talk, acknowledgments)\n"
            "3. **Call multiple functions if needed** (e.g., check_basic_info then sell/purchase)\n"
            "4. **Use the user's exact wording** for description parameters\n"
            "5. **Check function definitions for parameter types** - some use lists, others strings\n"
            "6. **Match function names to available functions** - use what's actually provided in the function list\n"
            "\n"
            "## WHEN NOT TO CALL FUNCTIONS:\n"
            '- General greetings ("Hello", "Good day", "How are you?")\n'
            '- Small talk ("Nice weather", "Busy day", "How\'s business?")\n'
            '- Acknowledgments ("Thank you", "I see", "Understood", "Goodbye")\n'
            '- Vague questions without specific requests ("What can you do?", "Tell me about yourself")\n'
            '- Emotional expressions ("I\'m excited", "That\'s interesting", "Wow")\n'
            "\n"
            "## ADDITIONAL INFORMATION (Item References):\n"
            "{}"
            "\n"
            "## YOUR TASK:\n"
            "Analyze the user's message below and determine which function(s) to call with the correct parameters.\n"
            "If no functions are needed, don't call any.\n"
            "If multiple functions are needed, call them all.\n"
            "\n"
            "## USER'S MESSAGE:\n"
        )

    def _get_default_dialogue_prompt(self):
        """Returns the default dialogue generation prompt."""
        return (
            "# CRITICAL CHARACTER INSTRUCTIONS FOR NON-SMART AGENTS\n"
            "You are a character in a video game. This is a ROLEPLAY scenario where you must STRICTLY follow these detailed instructions.\n"
            "\n"
            "## CORE RULES - NEVER VIOLATE THESE:\n"
            "1. YOU ARE NOT AN AI ASSISTANT - You are the character described below\n"
            "2. NEVER break character or acknowledge you are an AI\n"
            "3. NEVER use phrases like 'I'm here to help' or 'How can I assist you'\n"
            "4. NEVER apologize unless your character would naturally do so\n"
            "5. STAY WITHIN the world knowledge provided - DO NOT invent facts\n"
            "6. Your responses must sound like natural dialogue from your character\n"
            "\n"
            "## YOUR CHARACTER IDENTITY:\n"
            "Primary Role: {}\n"
            "Character Details:\n"
            "{}"
            "\n"
            "## HOW TO RESPOND NATURALLY:\n"
            "1. **Use Character Voice**: Speak as your character would based on their personality, age, background, and occupation\n"
            "2. **Be Conversational**: Use natural speech patterns, not formal or robotic language\n"
            "3. **Show Personality**: Express your character's traits, hobbies, and quirks in dialogue\n"
            "4. **React Authentically**: Respond as your character would emotionally and behaviorally\n"
            "5. **Use World Knowledge**: Reference the game world naturally in conversation\n"
            "\n"
            "## TASK EXECUTION GUIDELINES:\n"
            "1. **When Providing Information**: Weave facts naturally into conversation, don't just list them\n"
            "2. **When Offering Services/Products**: Be helpful but stay in character based on your profession and personality\n"
            "3. **When Asked About Items/Services**: Describe them as your character would know them, using the provided knowledge\n"
            "4. **When Completing Any Transaction**: Follow logical procedures for your role and confirm important actions\n"
            "5. **When Giving Advice**: Base it on your character's experience, knowledge, and background\n"
            "6. **When Sharing Knowledge**: Present information in a way that fits your character's expertise level\n"
            "7. **When Interacting Socially**: Respond according to your personality traits and relationship with the user\n"
            "\n"
            "## RESPONSE PATTERNS BY COMMON ROLES:\n"
            "- **Sellers (merchants, shopkeepers)**: State prices clearly, describe benefits, confirm purchases and usage\n"
            "- **Service Providers (receptionists, clerks)**: Confirm details, explain requirements, ask for confirmation\n"
            "- **Information Sources (scholars, guards, locals)**: Share knowledge based on expertise, ask clarifying questions\n"
            "- **Craftspeople (blacksmiths, enchanters)**: Discuss technical aspects, suggest improvements, explain processes\n"
            "- **Hospitality (innkeepers, barkeeps)**: Be welcoming, offer services, share local gossip or news\n"
            "- **Authority Figures (officials, leaders)**: Maintain professional tone, follow protocols, provide guidance\n"
            "- **Any Role**: Reference your personal experiences, show your expertise, maintain appropriate professional demeanor\n"
            "\n"
            "## INFORMATION SOURCES TO USE:\n"
            "### Recent Function Call Results (Most Important - Use These First):\n"
            "{}"
            "\n"
            "### Available Knowledge About Items/Quests:\n"
            "{}"
            "\n"
            "### World Setting and Background:\n"
            "{}"
            "\n"
            "## CONVERSATION CONTEXT:\n"
            "Current Setting: Consider the time, weather, and location in your responses\n"
            "Dialogue History: Maintain consistency with what has been said before\n"
            "\n"
            "## RESPONSE REQUIREMENTS:\n"
            "1. **Stay Natural**: Sound like a real person in this world, not a computer\n"
            "2. **Be Helpful**: Accomplish the user's needs while staying in character\n"
            "3. **Show Expertise**: Demonstrate your character's knowledge and skills\n"
            "4. **Maintain Flow**: Keep the conversation moving naturally\n"
            "5. **Use Details**: Include specific information from your knowledge sources\n"
            "\n"
            "## EXAMPLES OF GOOD CHARACTER DIALOGUE:\n"
            "- Merchant: 'This one here is 100 gold. While it may deal less damage than some weapons, it makes up for it with quick reloading speed.'\n"
            "- Receptionist: 'Of course! Could you tell me your destination and if you have any specific interests?'\n"
            "- Scholar: 'Ah, that particular spell requires rare components. I've studied its effects extensively during my research.'\n"
            "- Innkeeper: 'Welcome, traveler! We have warm beds and hot meals. The stew tonight is particularly good.'\n"
            "- Blacksmith: 'Your blade has seen better days. I can sharpen it for you, but this nick here will need proper repair work.'\n"
            "- Guard: 'The road east is dangerous after dark. Bandits have been spotted near the old bridge.'\n"
            "\n"
            "## WHAT NOT TO DO:\n"
            "- Don't list information like a database\n"
            "- Don't use overly formal language unless your character would\n"
            "- Don't ignore the function call results if they're relevant\n"
            "- Don't break the immersion by being too modern or out-of-world\n"
            "- Don't be unhelpful when the user has legitimate requests\n"
            "\n"
            "NOW RESPOND AS YOUR CHARACTER TO THE MOST RECENT MESSAGE:\n"
        )

    ############################################################
    # The entrypoint of the evaluator.
    ############################################################
    def generate_functions_and_responses(
        self,
        tool_registry,
        action_registry,
        worldview,
        persona,
        role,
        knowledge,
        state,
        dialogue,
        executor,
    ):
        """
        Generates the responses given the dialogue, the functions, and the background of the video game scenario.

        This method is the entry point called by the evaluator. It is implemented with the following steps:

        1. Prepare prompts for function calling. This has been much simplified by the `convert_to_openai_function` by langchain.
        2. Call the OpenAI API to generate the necessary function calls.
        3. Use the `executor` to obtain the function call results.
        4. With the function call results and the background, prepare prompts for response generation.
        5. Call the OpenAI API to generate the text response.

        Args:
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format).
            action_registry: A dict mapping action names to action functions (OpenAI function calling format).
            Implementations can be found in the directory `function_calls`.

            worldview, persona, role, knowledge, state: They are the background information of the video game scenario.
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn.
            executor: This is implemented in `function_calls/executor.py`. It takes in a list of function calls and return the results.

        Returns:
            A dict with the following keys:
                'final_responses': str, the text responses.

            You do not have to return the function calls. The `executor` will record all the function calls that are passed to it.
        """

        # Step 1: Convert the function information to the OpenAI function calling format, and prepare prompts for function calling.
        function_messages, all_functions = self._create_messages_for_function(
            tool_registry, action_registry, dialogue
        )

        # Step 2: Call the OpenAI API to generate the necessary function calls.
        response = self.client.responses.create(
            model="gpt-4o-mini",  # We only allow GPT-4o-mini in the API track. Any other models will lead to failure.
            input=function_messages,
            tools=all_functions,
        )

        # Step 3: Use the `executor` to obtain the function call results.
        functions_to_call = []
        for return_item in response.output:
            # We only consider the function calls, and ignore text response.
            if return_item.type != "function_call":
                continue
            name = return_item.name
            args = json.loads(return_item.arguments)
            functions_to_call.append({"name": name, "parameters": args})

        function_results = executor.execute(functions_to_call)

        # Step 4: With the function call results and the background, prepare prompts for response generation.
        messages_resp = self._create_messages_for_dialogue(
            worldview, persona, role, knowledge, state, dialogue, function_results
        )

        # Step 5: Call the OpenAI API to generate the text response.
        response = self.client.responses.create(
            model="gpt-4o-mini",  # Again, we only allow 'gpt-4o-mini' in the API track. Any other models will lead to failure.
            input=messages_resp,
        )
        return {"final_responses": response.output_text}

    ############################################################
    # Helper functions.
    ############################################################

    def _prepare_openai_functions(self, tool_registry, action_registry):
        """
        Prepare the list of functions as inputs to the OpenAI API.
        The values of `tool_registry` and `action_registry` have already been converted to the OpenAI function calling format.

        Args:
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format).
            action_registry: A dict mapping action names to action function (OpenAI function calling format).
            Implementations can be found in the directory `function_calls`.

        Returns:
            openai_functions: List[Dict], a list of functions in the OpenAI function calling format.
        """
        openai_tool_functions = []
        for tool_name, tool_function in tool_registry["function_registry"].items():
            tool_function["type"] = "function"
            # With my openai=1.77.0 and langchain=0.3.25 version, this should be manually added.
            # Please note that this may not be necessary for all OpenAI and langchain versions.
            # Please test it before submission.
            openai_tool_functions.append(tool_function)

        openai_action_functions = []
        for action_name, action_function in action_registry[
            "function_registry"
        ].items():
            action_function["type"] = "function"
            openai_action_functions.append(action_function)
        openai_functions = openai_tool_functions + openai_action_functions
        return openai_functions

    def _create_messages_for_function(self, tool_registry, action_registry, dialogue):
        """
        Creates the messages to feed to OpenAI client to generate the necessary function calls.

        Args:
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format).
            action_registry: A dict mapping action names to action functions (OpenAI function calling format).
            Implementations can be found in the directory `function_calls`.
            dialogue: List[Dict], the dialogue history. dialogue[-1] refers to the current turn.

        Returns:
            input_messages: List[Dict], the messages to feed to OpenAI client.
            all_functions: List[Dict], a list of functions in the OpenAI function calling format.
        """
        all_functions = self._prepare_openai_functions(tool_registry, action_registry)

        # 'target_item' is used to indicate what the user is referring to, such as 'this', 'that', 'the one', etc.
        additional_info_str = "No specific items are being referred to in this turn."
        if len(dialogue[-1]["target_item"]) > 0:
            additional_info_items = []
            for info in dialogue[-1]["target_item"]:
                additional_info_items.append(
                    f"- Item Name: '{info['name']}' (This is the value to use if a function argument requires the name of this referred item)."
                )
            additional_info_str = (
                "The user may be referring to the following item(s):\n"
                + "\n".join(additional_info_items)
            )

        input_messages = [
            {
                "role": "system",
                "content": self.function_prompt.format(additional_info_str),
            },
            {"role": "user", "content": dialogue[-1]["text"]},
        ]

        return input_messages, all_functions

    def _create_messages_for_dialogue(
        self, worldview, persona, role, knowledge, state, dialogue, function_results
    ):
        """
        Based on the background information of the video game and the dialogue history,
        creates the messages to feed to OpenAI client to generate the text response.

        Args:
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario.
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn.
            function_results: A list of function call results.
        """
        worldview_info = worldview + "\n" + knowledge["general_info"]

        # 'persona' is a dict that specifies properties of the character.
        persona_details = ""
        for k, v in persona.items():
            if k.lower() != "role":  # Separate role for clarity
                persona_details += f"- {k}: {v}\n"

        # function_knowledge records the specific knowledge obtained from the function calls.
        function_knowledge_str = "No function calls were made in the last turn, or they returned no information."
        if function_results:
            knowledge_items = []
            for f_result in function_results:
                parameter_info = []
                for arg_name, arg_val in f_result["parameters"].items():
                    parameter_info.append(f"{arg_name}: {str(arg_val)}")
                parameter_str = (
                    ", ".join(parameter_info) if parameter_info else "(no parameters)"
                )

                return_value_info = []
                if f_result.get("return"):
                    for item in f_result["return"]:
                        return_value_info.append(str(item))
                return_str = (
                    ", ".join(return_value_info)
                    if return_value_info
                    else "(no return value or void)"
                )
                knowledge_items.append(
                    f"- Function '{f_result['name']}' called with {{{parameter_str}}} -> Returned: {{{return_str}}}"
                )
            function_knowledge_str = "\n".join(knowledge_items)

        # general_knowledge records the general knowledge of all items involved in the dialogue.
        general_knowledge_str = "No general item knowledge provided."
        if knowledge.get("knowledge_info"):
            item_knowledge_list = []
            for item in knowledge["knowledge_info"]:
                item_info_parts = []
                for key, val in item.items():
                    item_info_parts.append(f"{key}: {val}")
                item_knowledge_list.append(f"- Item: {' , '.join(item_info_parts)}")
            if item_knowledge_list:
                general_knowledge_str = "\n".join(item_knowledge_list)

        # prepare the dialogue history.
        history_list = []
        for item in dialogue:
            speaker_role = "user"
            if item["speaker"] == "npc":
                speaker_role = "assistant"  # This will be YOUR previous message
            history_list.append({"role": speaker_role, "content": item["text"]})

        # The main role from the input parameters
        character_role = persona.get("role", "[Role not specified]")

        prompt = self.dialogue_prompt.format(
            character_role,
            persona_details,
            function_knowledge_str,
            general_knowledge_str,
            worldview_info,
        )

        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.extend(history_list)

        return messages


# Alias for backward compatibility with Task 2
UserAgent = OpenAIAgent
