from typing import List, Dict
import json
import litellm
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


class LMEvaluator:
    """Handles evaluation of generated responses and functions using language models via LiteLLM."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """
        Initialize the LMEvaluator with LiteLLM.

        Args:
            model: Model name to use (e.g., "gpt-4o-mini", "claude-3-sonnet", etc.)
            temperature: Temperature for generation
        """
        self.model = model
        self.temperature = temperature

        logger.info(f"🧠 Initializing LMEvaluator with model: {model}")
        logger.info(f"🌡️ Temperature: {temperature}")

        # Set up LiteLLM configuration
        if not os.getenv("OPENAI_API_KEY") and "gpt" in model.lower():
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI models"
            )

    def evaluate_with_lm(
        self,
        generated_response: str,
        gold_response: str,
        generated_functions: List[Dict],
        gold_functions: List[Dict],
        dialogue_context: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate the generated response and functions against gold standards using an LM.

        Returns:
            Dict containing scores for response and functions (0-10 scale)
        """
        try:
            logger.debug("📊 Starting LM evaluation")

            logger.debug("📝 Evaluating response quality")
            response_score = self._evaluate_response(
                generated_response, gold_response, dialogue_context
            )

            logger.debug("🛠️ Evaluating function calls")
            functions_score = self._evaluate_functions(
                generated_functions, gold_functions, dialogue_context
            )

            overall_score = (response_score + functions_score) / 2

            logger.debug(
                f"📊 Evaluation completed - Response: {response_score:.2f}, Functions: {functions_score:.2f}, Overall: {overall_score:.2f}"
            )

            return {
                "response_score": response_score,
                "functions_score": functions_score,
                "overall_score": overall_score,
            }

        except Exception as e:
            logger.error(f"❌ Error during evaluation: {e}")
            return {
                "response_score": 0.0,
                "functions_score": 0.0,
                "overall_score": 0.0,
                "error": str(e),
            }

    def _evaluate_response(
        self, generated_response: str, gold_response: str, dialogue_context: List[Dict]
    ) -> float:
        """Evaluate the generated response against gold response."""
        response_eval_prompt = f"""
You are an expert evaluator for dialogue systems in RPG/game scenarios. Your task is to evaluate how well a generated response matches the expected gold response.

CONTEXT:
The following is a dialogue exchange in an RPG game scenario where the player is interacting with an NPC.

DIALOGUE CONTEXT:
{json.dumps(dialogue_context, indent=2)}

EVALUATION TASK:
Compare the GENERATED response against the GOLD response and score how well they match in terms of:
1. Content accuracy (does it convey the same information?)
2. Tone and style (does it match the character's personality?)
3. Contextual appropriateness (does it fit the game scenario?)
4. Completeness (does it address all necessary points?)

GENERATED RESPONSE:
{generated_response}

GOLD RESPONSE:
{gold_response}

Please provide a score from 0-10 where:
- 0-2: Poor match, significant differences in content or tone
- 3-4: Below average, some key differences
- 5-6: Average, mostly similar but with notable differences
- 7-8: Good match, minor differences only
- 9-10: Excellent match, essentially equivalent

Respond with just the numeric score (e.g., "7.5").
"""

        logger.debug("🔍 Sending response evaluation to LM")
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": response_eval_prompt}],
            temperature=self.temperature,
        )
        score = float(response.choices[0].message.content.strip())
        logger.debug(f"📝 Response evaluation score: {score}")
        return score

    def _evaluate_functions(
        self,
        generated_functions: List[Dict],
        gold_functions: List[Dict],
        dialogue_context: List[Dict],
    ) -> float:
        """Evaluate the generated functions against gold functions."""

        # If no functions are expected or generated, perfect score
        if not generated_functions and not gold_functions:
            logger.debug("🛠️ No functions expected or generated - perfect score")
            return 10.0

        functions_eval_prompt = f"""
You are an expert evaluator for function calling systems in RPG/game scenarios. Your task is to evaluate how well generated function calls match the expected gold function calls.

DIALOGUE CONTEXT:
{json.dumps(dialogue_context, indent=2)}

EVALUATION TASK:
Compare the GENERATED function calls against the GOLD function calls and score how well they match in terms of:
1. Function selection (are the correct functions called?)
2. Parameter accuracy (are the parameters correct?)
3. Completeness (are all necessary function calls present?)
4. Order and timing (are functions called in appropriate sequence?)

GENERATED FUNCTIONS:
{json.dumps(generated_functions, indent=2)}

GOLD FUNCTIONS:
{json.dumps(gold_functions, indent=2)}

Please provide a score from 0-10 where:
- 0-2: Poor match, wrong functions or incorrect parameters
- 3-4: Below average, some correct functions but significant issues
- 5-6: Average, mostly correct but with notable differences
- 7-8: Good match, minor differences only
- 9-10: Excellent match, essentially equivalent

If no functions are expected or generated, respond with "10.0".

Respond with just the numeric score (e.g., "8.0").
"""

        logger.debug("🔍 Sending function evaluation to LM")
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": functions_eval_prompt}],
            temperature=self.temperature,
        )
        score = float(response.choices[0].message.content.strip())
        logger.debug(f"🛠️ Function evaluation score: {score}")
        return score

    def evaluate_batch_parallel(
        self, evaluation_batch: List[Dict], n_parallel: int = 16
    ) -> List[Dict]:
        """
        Evaluate a batch of turns in parallel using semaphore pattern.

        Args:
            evaluation_batch: List of evaluation data containing:
                - generated_response
                - gold_response
                - generated_functions
                - gold_functions
                - dialogue_context
            n_parallel: Number of parallel evaluation requests

        Returns:
            List of evaluation results
        """
        if not evaluation_batch:
            return []

        logger.info(
            f"🚀 Starting parallel evaluation of {len(evaluation_batch)} turns with {n_parallel} workers"
        )

        # Use asyncio to run the parallel evaluation
        try:
            results = asyncio.run(
                self._evaluate_batch_async(evaluation_batch, n_parallel)
            )
            logger.success(
                f"✅ Completed parallel evaluation of {len(evaluation_batch)} turns"
            )
            return results
        except Exception as e:
            logger.error(f"❌ Error in parallel evaluation: {e}")
            # Fallback to sequential evaluation
            logger.warning("🔄 Falling back to sequential evaluation")
            return [
                self.evaluate_with_lm(**batch_item) for batch_item in evaluation_batch
            ]

    async def _evaluate_batch_async(
        self, evaluation_batch: List[Dict], n_parallel: int
    ) -> List[Dict]:
        """Async implementation of batch evaluation with semaphore."""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(n_parallel)

        async def evaluate_single_with_semaphore(batch_item: Dict) -> Dict:
            """Evaluate a single item with semaphore control."""
            async with semaphore:
                # Run the synchronous evaluation in thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(
                        executor,
                        self.evaluate_with_lm,
                        batch_item["generated_response"],
                        batch_item["gold_response"],
                        batch_item["generated_functions"],
                        batch_item["gold_functions"],
                        batch_item["dialogue_context"],
                    )
                return result

        # Create tasks for all evaluations
        tasks = [
            evaluate_single_with_semaphore(batch_item)
            for batch_item in evaluation_batch
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error evaluating item {i}: {result}")
                processed_results.append(
                    {
                        "response_score": 0.0,
                        "functions_score": 0.0,
                        "overall_score": 0.0,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results
