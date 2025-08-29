from typing import Dict, Optional
import copy
from loguru import logger

from agents import OpenAIAgent
from .lm_evaluator import LMEvaluator


class ConversationProcessor:
    """Handles processing of conversations and generating responses."""

    def __init__(self, agent: OpenAIAgent, evaluator: Optional[LMEvaluator] = None):
        self.agent = agent
        self.evaluator = evaluator or LMEvaluator()
        logger.debug("🗣️ ConversationProcessor initialized")

    def get_functions_and_responses(
        self, cur_conv, cur_turn, tool_registry, action_registry, executor
    ) -> str:
        """
        Generate functions and responses for a conversation turn.

        Returns:
            Generated response string
        """
        logger.debug("🔧 Generating functions and responses")

        dialogue = [
            {"speaker": msg.speaker, "text": msg.text, "target_item": msg.target_items}
            for msg in cur_turn.messages
        ]

        logger.debug(f"📝 Processing {len(dialogue)} dialogue messages")

        # Check if this is the Task 2 agent method (more parameters)
        if hasattr(self.agent, "generate_functions_and_responses"):
            logger.debug("🎮 Using Task 2 agent interface")
            all_results = self.agent.generate_functions_and_responses(
                tool_registry,
                action_registry,
                cur_conv.worldview,
                cur_conv.personas["npc"].to_dict(),
                cur_conv.roles["npc"],
                {
                    "general_info": cur_conv.general_knowledge,
                    "knowledge_info": cur_conv.knowledge,
                },
                cur_conv.state,
                dialogue,
                executor,
            )
        else:
            # Fallback for simpler agent interface
            logger.debug("🎮 Using fallback agent interface")
            all_results = self.agent.generate_response(
                dialogue,
                executor,
            )

        response = all_results["final_responses"]
        logger.debug(f"✅ Generated response ({len(response)} chars)")
        return response

    def process_single_conversation(
        self, conversation, enable_evaluation: bool = True
    ) -> tuple[Dict, Dict]:
        """
        Process a single conversation and return responses and evaluations.

        Returns:
            Tuple of (responses_dict, evaluations_dict)
        """
        logger.debug(f"🗣️ Processing conversation with {len(conversation.turns)} turns")
        cur_conv_responses = {}
        cur_conv_evaluations = {}

        # Import function maps
        from function_calls import tool_map, action_map, Executor

        tool_registry = tool_map[conversation.function_list_id]
        action_registry = action_map[conversation.function_list_id]

        logger.debug(
            f"🛠️ Loaded function registries for ID: {conversation.function_list_id}"
        )

        for turn_idx, turn in enumerate(conversation.turns):
            logger.debug(f"🔄 Processing turn {turn_idx + 1}/{len(conversation.turns)}")

            gold_functions = [
                {
                    "name": function.name,
                    "parameters": function.parameters,
                    "return": function.return_values,
                }
                for function in turn.gold_functions
            ]

            logger.debug(f"📋 Turn has {len(gold_functions)} gold functions")
            cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)

            response = self.get_functions_and_responses(
                conversation, turn, tool_registry, action_registry, cur_turn_exec
            )

            # Store generated response and functions
            function_calls = copy.deepcopy(cur_turn_exec.function_call_stats)
            cur_conv_responses[f"turn_{turn_idx}"] = {
                "response": response,
                "functions": function_calls,
            }

            logger.debug(f"🛠️ Turn generated {len(function_calls)} function calls")

            # Perform evaluation if enabled and evaluator is available
            if enable_evaluation and self.evaluator:
                logger.debug("📊 Starting evaluation for turn")

                dialogue_context = [
                    {
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "target_item": msg.target_items,
                    }
                    for msg in turn.messages
                ]

                evaluation_scores = self.evaluator.evaluate_with_lm(
                    response,
                    turn.gold_response,
                    function_calls,
                    gold_functions,
                    dialogue_context,
                )

                cur_conv_evaluations[f"turn_{turn_idx}"] = evaluation_scores

                if "error" not in evaluation_scores:
                    logger.debug(
                        f"📊 Turn evaluation - Overall: {evaluation_scores['overall_score']:.2f}"
                    )
                else:
                    logger.warning(
                        f"⚠️ Evaluation error: {evaluation_scores.get('error', 'Unknown')}"
                    )

        logger.debug("✅ Conversation processing completed")
        return cur_conv_responses, cur_conv_evaluations

    def process_conversations_parallel(
        self, conversations, enable_evaluation: bool = True, n_parallel: int = 16
    ) -> tuple[list, list]:
        """
        Process multiple conversations with parallel evaluation.

        Args:
            conversations: List of conversation objects
            enable_evaluation: Whether to enable evaluation
            n_parallel: Number of parallel evaluation requests

        Returns:
            Tuple of (responses_list, evaluations_list)
        """
        logger.info(
            f"🚀 Processing {len(conversations)} conversations with parallel evaluation"
        )

        all_responses = []
        all_evaluations = []

        # First, generate all responses sequentially (this is typically fast)
        evaluation_batch = []

        for conv_idx, conversation in enumerate(conversations):
            logger.debug(
                f"🗣️ Processing conversation {conv_idx + 1}/{len(conversations)}"
            )

            cur_conv_responses = {}

            # Import function maps
            from function_calls import tool_map, action_map, Executor

            tool_registry = tool_map[conversation.function_list_id]
            action_registry = action_map[conversation.function_list_id]

            for turn_idx, turn in enumerate(conversation.turns):
                gold_functions = [
                    {
                        "name": function.name,
                        "parameters": function.parameters,
                        "return": function.return_values,
                    }
                    for function in turn.gold_functions
                ]

                cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)

                response = self.get_functions_and_responses(
                    conversation, turn, tool_registry, action_registry, cur_turn_exec
                )

                function_calls = copy.deepcopy(cur_turn_exec.function_call_stats)
                cur_conv_responses[f"turn_{turn_idx}"] = {
                    "response": response,
                    "functions": function_calls,
                }

                # Prepare evaluation data for parallel processing
                if enable_evaluation and self.evaluator:
                    dialogue_context = [
                        {
                            "speaker": msg.speaker,
                            "text": msg.text,
                            "target_item": msg.target_items,
                        }
                        for msg in turn.messages
                    ]

                    evaluation_batch.append(
                        {
                            "conversation_idx": conv_idx,
                            "turn_idx": turn_idx,
                            "generated_response": response,
                            "gold_response": turn.gold_response,
                            "generated_functions": function_calls,
                            "gold_functions": gold_functions,
                            "dialogue_context": dialogue_context,
                        }
                    )

            all_responses.append(cur_conv_responses)

        # Now perform parallel evaluation of all turns
        if enable_evaluation and self.evaluator and evaluation_batch:
            logger.info(
                f"📊 Starting parallel evaluation of {len(evaluation_batch)} turns"
            )

            evaluation_results = self.evaluator.evaluate_batch_parallel(
                evaluation_batch, n_parallel
            )

            # Organize evaluation results back into conversation structure
            for conv_idx in range(len(conversations)):
                all_evaluations.append({})

            for eval_data, eval_result in zip(evaluation_batch, evaluation_results):
                conv_idx = eval_data["conversation_idx"]
                turn_idx = eval_data["turn_idx"]
                all_evaluations[conv_idx][f"turn_{turn_idx}"] = eval_result

            logger.success(
                f"✅ Completed parallel evaluation of {len(evaluation_batch)} turns"
            )
        else:
            # No evaluation requested
            all_evaluations = [{} for _ in conversations]

        return all_responses, all_evaluations
