from typing import List, Dict, Optional
import json
import time
import os
from progiter import ProgIter
from loguru import logger
from agents import OpenAIAgent
from .data_loader import DataLoader
from .lm_evaluator import LMEvaluator
from .conversation_processor import ConversationProcessor


class Task2Runner:
    """Main orchestrator class for running Task 2 processing."""

    def __init__(
        self, evaluation_model: str = "gpt-4o-mini", evaluation_temperature: float = 0.1
    ):
        """
        Initialize the Task2Runner.

        Args:
            evaluation_model: Model to use for evaluation (e.g., "gpt-4o-mini", "claude-3-sonnet")
            evaluation_temperature: Temperature for evaluation model
        """
        logger.info("🏗️ Initializing Task2Runner")
        logger.info(f"🧠 Evaluation model: {evaluation_model}")
        logger.info(f"🌡️ Evaluation temperature: {evaluation_temperature}")

        self.data_loader = DataLoader()
        self.agent = OpenAIAgent()
        self.evaluator = LMEvaluator(
            model=evaluation_model, temperature=evaluation_temperature
        )
        self.processor = ConversationProcessor(self.agent, self.evaluator)

        logger.success("✅ Task2Runner initialized successfully")

    def process_conversations(
        self, data_path: str, save_path: str, enable_evaluation: bool = True
    ) -> None:
        """Process all conversations and save results."""
        logger.info("📋 Starting conversation processing")
        start_time = time.time()

        logger.info(f"📥 Loading data from: {data_path}")
        data_set = self.data_loader.load_data(data_path)
        logger.info(f"📊 Loaded {len(data_set)} conversations")

        # Ensure save directory exists
        save_directory = os.path.dirname(save_path)
        if not os.path.exists(save_directory):
            logger.info(f"📁 Creating save directory: {save_directory}")
            os.makedirs(save_directory)

        generated_responses = []
        all_evaluation_scores = []

        logger.info("🔄 Processing conversations...")
        for conv_idx, conversation in ProgIter(
            enumerate(data_set), desc="Processing conversations", verbose=2
        ):
            logger.debug(f"🗣️ Processing conversation {conv_idx + 1}/{len(data_set)}")

            responses, evaluations = self.processor.process_single_conversation(
                conversation, enable_evaluation
            )

            generated_responses.append(responses)
            if enable_evaluation:
                all_evaluation_scores.append(evaluations)

            logger.debug(f"✅ Completed conversation {conv_idx + 1}")

        # Save results
        logger.info("💾 Saving results...")
        self._save_results(
            generated_responses, all_evaluation_scores, save_path, enable_evaluation
        )

        total_time = time.time() - start_time
        logger.success(f"💾 Results saved to: {save_path}")
        logger.info(f"⏱️ Total processing time: {total_time:.2f} seconds")
        logger.info(
            f"📈 Average time per conversation: {total_time / len(data_set):.2f} seconds"
        )

    def _save_results(
        self,
        generated_responses: List[Dict],
        all_evaluation_scores: List[Dict],
        save_path: str,
        enable_evaluation: bool,
    ) -> None:
        """Save results to file with optional evaluation summary."""
        results = {"generated_responses": generated_responses}

        if enable_evaluation and all_evaluation_scores:
            results["evaluations"] = all_evaluation_scores

            # Calculate average scores
            summary = self._calculate_evaluation_summary(all_evaluation_scores)
            if summary:
                results["summary"] = summary
                logger.info("📊 Evaluation Summary:")
                logger.info(
                    f"  📝 Average Response Score: {summary['avg_response_score']:.3f}"
                )
                logger.info(
                    f"  🛠️ Average Functions Score: {summary['avg_functions_score']:.3f}"
                )
                logger.info(
                    f"  🎯 Average Overall Score: {summary['avg_overall_score']:.3f}"
                )
                logger.info(
                    f"  📋 Total Evaluated Turns: {summary['total_evaluated_turns']}"
                )

        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)

    def _calculate_evaluation_summary(
        self, all_evaluation_scores: List[Dict]
    ) -> Optional[Dict]:
        """Calculate summary statistics for evaluation scores."""
        total_response_score = 0
        total_functions_score = 0
        total_overall_score = 0
        total_turns = 0

        for conv_eval in all_evaluation_scores:
            for turn_eval in conv_eval.values():
                if "error" not in turn_eval:
                    total_response_score += turn_eval["response_score"]
                    total_functions_score += turn_eval["functions_score"]
                    total_overall_score += turn_eval["overall_score"]
                    total_turns += 1

        if total_turns > 0:
            logger.debug(f"📊 Calculated summary for {total_turns} turns")
            return {
                "avg_response_score": total_response_score / total_turns,
                "avg_functions_score": total_functions_score / total_turns,
                "avg_overall_score": total_overall_score / total_turns,
                "total_evaluated_turns": total_turns,
            }
        logger.warning("⚠️ No valid evaluation scores found")
        return None
