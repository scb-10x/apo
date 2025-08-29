import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from loguru import logger

from agents import OpenAIAgent
from tasks.task1_runner import Task1Runner
from tasks.task2_runner import Task2Runner
from tasks.data_loader import DataLoader
from optimization.gradient_generator import GradientGenerator
from optimization.prompt_editor import PromptEditor
from optimization.config import (
    OptimizationConfig,
    OptimizationState,
    IterationResult,
    PromptCandidate,
    GradientMemory,
)


class PromptOptimizer:
    """Prompt optimizer for testing and basic optimization."""

    def __init__(self, config: OptimizationConfig):
        """Initialize the optimizer."""
        self.config = config
        self.state = OptimizationState()

        # Initialize components
        self.gradient_generator = GradientGenerator(
            model=config.gradient_model,
            temperature=config.gradient_temperature,
            score_threshold=config.score_threshold,
        )

        self.prompt_editor = PromptEditor(
            model=config.editor_model,
            temperature=config.editor_temperature,
            edit_strategy=config.prompt_edit_strategy,
        )

        # Initialize task runners
        self.task1_runner = (
            Task1Runner(
                evaluation_model=config.evaluator_model,
                evaluation_temperature=config.evaluator_temperature,
            )
            if config.task in ["task1", "both"]
            else None
        )

        self.task2_runner = (
            Task2Runner(
                evaluation_model=config.evaluator_model,
                evaluation_temperature=config.evaluator_temperature,
            )
            if config.task in ["task2", "both"]
            else None
        )

        logger.info("🚀 Initialized SimplePromptOptimizer")

    def optimize(self) -> OptimizationState:
        """Run the optimization loop."""
        logger.info("🚀 Starting prompt optimization")

        self._initialize_optimization()

        try:
            # Main optimization loop
            while not self._should_stop():
                self._run_iteration()

            logger.info("🏁 Optimization completed")
            self._save_results()

            return self.state

        except KeyboardInterrupt:
            logger.warning("⚠️ Optimization interrupted by user")
            raise
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise

    def _initialize_optimization(self) -> None:
        """Initialize optimization state."""
        logger.info("🔧 Initializing optimization state")

        # Set start time
        self.state.start_time = datetime.now().isoformat()

        # Get initial prompts from agent
        agent = OpenAIAgent()

        if "function" in self.config.prompt_types:
            self.state.current_function_prompt = (
                self.config.initial_function_prompt
                or agent._get_default_function_prompt()
            )
            self.state.best_function_prompt = self.state.current_function_prompt

        if "dialogue" in self.config.prompt_types:
            self.state.current_dialogue_prompt = (
                self.config.initial_dialogue_prompt
                or agent._get_default_dialogue_prompt()
            )
            self.state.best_dialogue_prompt = self.state.current_dialogue_prompt

        # Evaluate initial performance
        logger.info("📊 Evaluating initial prompt performance")
        initial_score, initial_details = self._evaluate_prompts(
            self.state.current_function_prompt, self.state.current_dialogue_prompt
        )

        # Record initial iteration
        initial_result = IterationResult(
            iteration=0,
            function_prompt=self.state.current_function_prompt,
            dialogue_prompt=self.state.current_dialogue_prompt,
            score=initial_score,
            response_score=initial_details.get("response_score", 0.0),
            functions_score=initial_details.get("functions_score", 0.0),
            total_samples=initial_details.get("total_samples", 0),
            failed_samples=initial_details.get("failed_samples", 0),
            improvement=0.0,
            timestamp=datetime.now().isoformat(),
            gradient_count=0,
        )

        self.state.iteration_results.append(initial_result)
        self.state.best_score = initial_score
        self.state.total_evaluations += 1

        # Initialize beam search if enabled
        self._initialize_beam_search()

        logger.info(f"📊 Initial performance: {initial_score:.3f}")

    def _run_iteration(self) -> None:
        """Run a single optimization iteration."""
        self.state.iteration += 1
        logger.info(f"🔄 Starting iteration {self.state.iteration}")

        # Get evaluation results for gradient generation
        evaluation_results = self._get_detailed_evaluation_results(
            self.state.current_function_prompt, self.state.current_dialogue_prompt
        )

        # Generate gradients from failed samples with mini-batch support
        mini_batch_size = (
            self.config.gradient_mini_batch_size
            if self.config.enable_gradient_mini_batch
            else None
        )

        gradients = self.gradient_generator.generate_gradients(
            evaluation_results,
            max_samples=self.config.max_gradient_samples,
            mini_batch_size=mini_batch_size,
        )

        if not gradients:
            logger.warning("⚠️ No gradients generated - skipping prompt editing")
            # Create a dummy iteration result to maintain the loop
            self._create_dummy_iteration_result()
            return

        # Summarize gradients
        gradient_summary = self.gradient_generator.summarize_gradients(gradients)

        # Update gradient memory
        self._update_gradient_memory(gradients, gradient_summary)

        # Use beam search if enabled
        if self.config.enable_beam_search:
            self._beam_search_step(gradients, gradient_summary)
            # Beam search updates current prompts internally
            new_function_prompt = self.state.current_function_prompt
            new_dialogue_prompt = self.state.current_dialogue_prompt
        else:
            # Standard prompt editing with optional candidate generation
            new_function_prompt = self.state.current_function_prompt
            new_dialogue_prompt = self.state.current_dialogue_prompt

            # Edit function prompt
            if "function" in self.config.prompt_types and gradients:
                if self.config.enable_prompt_candidates:
                    # Generate multiple candidates and select best via Monte Carlo
                    candidates = self.prompt_editor.generate_prompt_candidates(
                        current_prompt=self.state.current_function_prompt,
                        gradients=gradients,
                        gradient_summary=gradient_summary,
                        prompt_type="function",
                        num_candidates=self.config.num_prompt_candidates,
                        gradient_memory=self.state.gradient_memory
                        if self.config.enable_gradient_memory
                        else None,
                    )

                    if candidates:
                        new_function_prompt, _ = self._monte_carlo_evaluate_candidates(
                            candidates, "function"
                        )
                        logger.info(
                            f"🎯 Selected function prompt from {len(candidates)} candidates"
                        )
                else:
                    # Standard single prompt editing
                    new_function_prompt = self.prompt_editor.edit_function_prompt(
                        self.state.current_function_prompt, gradients, gradient_summary
                    )

            # Edit dialogue prompt
            if "dialogue" in self.config.prompt_types and gradients:
                if self.config.enable_prompt_candidates:
                    # Generate multiple candidates and select best via Monte Carlo
                    candidates = self.prompt_editor.generate_prompt_candidates(
                        current_prompt=self.state.current_dialogue_prompt,
                        gradients=gradients,
                        gradient_summary=gradient_summary,
                        prompt_type="dialogue",
                        num_candidates=self.config.num_prompt_candidates,
                        gradient_memory=self.state.gradient_memory
                        if self.config.enable_gradient_memory
                        else None,
                    )

                    if candidates:
                        new_dialogue_prompt, _ = self._monte_carlo_evaluate_candidates(
                            candidates, "dialogue"
                        )
                        logger.info(
                            f"🎯 Selected dialogue prompt from {len(candidates)} candidates"
                        )
                else:
                    # Standard single prompt editing
                    new_dialogue_prompt = self.prompt_editor.edit_dialogue_prompt(
                        self.state.current_dialogue_prompt, gradients, gradient_summary
                    )

        # Evaluate new prompts (full evaluation, not sample)
        new_score, new_details = self._evaluate_prompts(
            new_function_prompt, new_dialogue_prompt
        )

        # Calculate improvement
        previous_score = (
            self.state.iteration_results[-1].score
            if self.state.iteration_results
            else 0.0
        )
        improvement = new_score - previous_score

        # Record iteration results
        iteration_result = IterationResult(
            iteration=self.state.iteration,
            function_prompt=new_function_prompt,
            dialogue_prompt=new_dialogue_prompt,
            score=new_score,
            response_score=new_details.get("response_score", 0.0),
            functions_score=new_details.get("functions_score", 0.0),
            total_samples=new_details.get("total_samples", 0),
            failed_samples=new_details.get("failed_samples", 0),
            improvement=improvement,
            timestamp=datetime.now().isoformat(),
            gradient_count=len(gradients),
        )

        self.state.iteration_results.append(iteration_result)
        self.state.total_evaluations += 1

        # Update current prompts and best score
        self.state.current_function_prompt = new_function_prompt
        self.state.current_dialogue_prompt = new_dialogue_prompt

        if new_score > self.state.best_score:
            self.state.best_score = new_score
            self.state.best_function_prompt = new_function_prompt
            self.state.best_dialogue_prompt = new_dialogue_prompt
            logger.success(
                f"🎯 New best score: {new_score:.3f} (improvement: +{improvement:.3f})"
            )
        else:
            logger.info(f"📊 Score: {new_score:.3f} (change: {improvement:+.3f})")

    def _create_dummy_iteration_result(self) -> None:
        """Create a dummy iteration result when no gradients are generated."""
        previous_result = (
            self.state.iteration_results[-1] if self.state.iteration_results else None
        )

        if previous_result:
            dummy_result = IterationResult(
                iteration=self.state.iteration,
                function_prompt=previous_result.function_prompt,
                dialogue_prompt=previous_result.dialogue_prompt,
                score=previous_result.score,
                response_score=previous_result.response_score,
                functions_score=previous_result.functions_score,
                total_samples=previous_result.total_samples,
                failed_samples=previous_result.failed_samples,
                improvement=0.0,
                timestamp=datetime.now().isoformat(),
                gradient_count=0,
            )
            self.state.iteration_results.append(dummy_result)

    def _evaluate_prompts(
        self, function_prompt: Optional[str], dialogue_prompt: Optional[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate prompts and return overall score and details."""
        logger.debug("📊 Evaluating prompts...")

        # Create agent with custom prompts
        agent = OpenAIAgent(
            function_prompt=function_prompt, dialogue_prompt=dialogue_prompt
        )

        total_scores = []
        total_response_scores = []
        total_function_scores = []
        total_samples = 0
        total_failed = 0

        # Evaluate on task1 if specified
        if self.config.task in ["task1", "both"] and self.task1_runner:
            task1_scores = self._evaluate_on_task(self.task1_runner, agent, "task1")
            total_scores.extend(task1_scores["overall_scores"])
            total_response_scores.extend(task1_scores["response_scores"])
            total_function_scores.extend(task1_scores["function_scores"])
            total_samples += task1_scores["total_samples"]
            total_failed += task1_scores["failed_samples"]

        # Evaluate on task2 if specified
        if self.config.task in ["task2", "both"] and self.task2_runner:
            task2_scores = self._evaluate_on_task(self.task2_runner, agent, "task2")
            total_scores.extend(task2_scores["overall_scores"])
            total_response_scores.extend(task2_scores["response_scores"])
            total_function_scores.extend(task2_scores["function_scores"])
            total_samples += task2_scores["total_samples"]
            total_failed += task2_scores["failed_samples"]

        # Calculate average scores (already normalized in _evaluate_on_task)
        overall_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        avg_response_score = (
            sum(total_response_scores) / len(total_response_scores)
            if total_response_scores
            else 0.0
        )
        avg_function_score = (
            sum(total_function_scores) / len(total_function_scores)
            if total_function_scores
            else 0.0
        )

        return overall_score, {
            "response_score": avg_response_score,
            "functions_score": avg_function_score,
            "total_samples": total_samples,
            "failed_samples": total_failed,
        }

    def _evaluate_on_task(self, runner, agent, task_name: str) -> Dict[str, Any]:
        """Evaluate agent on a specific task."""
        logger.debug(f"🎯 Evaluating on {task_name}")

        # Replace the runner's agent with our custom agent
        original_agent = runner.agent
        runner.agent = agent

        try:
            # Load data
            data_loader = DataLoader()
            dataset = data_loader.load_data(self.config.data_path)

            # Limit sample size if specified
            if self.config.sample_size and len(dataset) > self.config.sample_size:
                # Ensure dataset is a proper list
                if not isinstance(dataset, list):
                    logger.debug(f"Converting dataset from {type(dataset)} to list")
                    dataset = list(dataset) if hasattr(dataset, "__iter__") else []

                if len(dataset) > 0:
                    dataset = random.sample(dataset, self.config.sample_size)

                # Process conversations and collect scores using parallel evaluation
            overall_scores = []
            response_scores = []
            function_scores = []
            failed_count = 0

            # Use parallel processing for better performance
            responses_list, evaluations_list = (
                runner.processor.process_conversations_parallel(
                    dataset, enable_evaluation=True, n_parallel=self.config.n_parallel
                )
            )

            # Extract scores from evaluations
            for evaluations in evaluations_list:
                for turn_key, turn_eval in evaluations.items():
                    if turn_key.startswith("turn_") and "error" not in turn_eval:
                        # Normalize scores from 0-10 to 0-1 scale
                        normalized_overall = turn_eval["overall_score"] / 10.0
                        normalized_response = turn_eval["response_score"] / 10.0
                        normalized_function = turn_eval["functions_score"] / 10.0

                        overall_scores.append(normalized_overall)
                        response_scores.append(normalized_response)
                        function_scores.append(normalized_function)

                        if normalized_overall < self.config.score_threshold:
                            failed_count += 1

            return {
                "overall_scores": overall_scores,
                "response_scores": response_scores,
                "function_scores": function_scores,
                "total_samples": len(overall_scores),
                "failed_samples": failed_count,
            }

        finally:
            # Restore original agent
            runner.agent = original_agent

    def _get_detailed_evaluation_results(
        self, function_prompt: Optional[str], dialogue_prompt: Optional[str]
    ) -> List[Dict]:
        """Get detailed evaluation results for gradient generation."""
        logger.debug("📊 Getting detailed evaluation results...")

        # Create agent with custom prompts
        agent = OpenAIAgent(
            function_prompt=function_prompt, dialogue_prompt=dialogue_prompt
        )

        all_results = []

        # Get results from task1 if specified
        if self.config.task in ["task1", "both"] and self.task1_runner:
            task1_results = self._get_task_evaluation_results(self.task1_runner, agent)
            all_results.extend(task1_results)

        # Get results from task2 if specified
        if self.config.task in ["task2", "both"] and self.task2_runner:
            task2_results = self._get_task_evaluation_results(self.task2_runner, agent)
            all_results.extend(task2_results)

        return all_results

    def _get_task_evaluation_results(self, runner, agent) -> List[Dict]:
        """Get detailed evaluation results from a specific task runner."""
        # Replace the runner's agent with our custom agent
        original_agent = runner.agent
        runner.agent = agent

        try:
            # Load data
            data_loader = DataLoader()
            dataset = data_loader.load_data(self.config.data_path)

            # Limit sample size if specified
            if self.config.sample_size and len(dataset) > self.config.sample_size:
                # Ensure dataset is a proper list
                if not isinstance(dataset, list):
                    logger.debug(f"Converting dataset from {type(dataset)} to list")
                    dataset = list(dataset) if hasattr(dataset, "__iter__") else []

                if len(dataset) > 0:
                    dataset = random.sample(dataset, self.config.sample_size)

            # Process conversations and collect detailed results using parallel evaluation
            responses_list, evaluations_list = (
                runner.processor.process_conversations_parallel(
                    dataset, enable_evaluation=True, n_parallel=self.config.n_parallel
                )
            )

            return evaluations_list

        finally:
            # Restore original agent
            runner.agent = original_agent

    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        # Check max iterations
        if self.state.iteration >= self.config.max_iterations:
            logger.info(f"🛑 Reached maximum iterations ({self.config.max_iterations})")
            return True

        # Check if threshold is reached
        if self.state.best_score >= self.config.score_threshold:
            logger.success(
                f"🎯 Reached score threshold ({self.config.score_threshold})"
            )
            return True

        # Check for convergence (no improvement for several iterations)
        if len(self.state.iteration_results) >= 3:
            recent_improvements = [
                r.improvement for r in self.state.iteration_results[-3:]
            ]
            if all(
                imp < self.config.min_improvement_threshold
                for imp in recent_improvements
            ):
                logger.info(
                    "📈 Convergence detected - minimal improvement in recent iterations"
                )
                return True

        return False

    def _monte_carlo_evaluate_candidates(
        self, candidates: List[str], prompt_type: str
    ) -> Tuple[str, float]:
        """
        Evaluate prompt candidates using Monte Carlo sampling.

        Args:
            candidates: List of candidate prompts
            prompt_type: Type of prompt ("function" or "dialogue")

        Returns:
            Tuple of (best_candidate, best_score)
        """
        logger.info(f"🎲 Monte Carlo evaluation of {len(candidates)} candidates")

        best_candidate = candidates[0] if candidates else ""
        best_score = 0.0

        for i, candidate in enumerate(candidates):
            logger.debug(f"🔄 Evaluating candidate {i + 1}/{len(candidates)}")

            # Set up temporary prompts for evaluation
            if prompt_type == "function":
                temp_function_prompt = candidate
                temp_dialogue_prompt = self.state.current_dialogue_prompt
            else:
                temp_function_prompt = self.state.current_function_prompt
                temp_dialogue_prompt = candidate

            # Evaluate with small sample size
            score, _ = self._evaluate_prompts_sample(
                temp_function_prompt,
                temp_dialogue_prompt,
                sample_size=self.config.monte_carlo_sample_size,
            )

            logger.debug(f"📊 Candidate {i + 1} score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_candidate = candidate

        logger.success(f"🏆 Best candidate score: {best_score:.3f}")
        return best_candidate, best_score

    def _evaluate_prompts_sample(
        self,
        function_prompt: Optional[str],
        dialogue_prompt: Optional[str],
        sample_size: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate prompts with a limited sample size.

        Args:
            function_prompt: Function calling prompt to evaluate
            dialogue_prompt: Dialogue generation prompt to evaluate
            sample_size: Number of samples to use for evaluation

        Returns:
            Tuple of (overall_score, evaluation_details)
        """
        logger.debug(f"📊 Evaluating prompts with sample size: {sample_size}")

        # Create agent with the prompts
        agent = OpenAIAgent(
            function_prompt=function_prompt, dialogue_prompt=dialogue_prompt
        )

        # Load data
        data_loader = DataLoader()
        dataset = data_loader.load_data(self.config.data_path)

        # Limit to sample size
        if len(dataset) > sample_size:
            # Ensure dataset is a proper list
            if not isinstance(dataset, list):
                logger.debug(f"Converting dataset from {type(dataset)} to list")
                dataset = list(dataset) if hasattr(dataset, "__iter__") else []

            if len(dataset) > 0:
                dataset = random.sample(dataset, sample_size)

        all_scores = []
        total_evaluations = 0

        # Evaluate on task1 if specified
        if self.config.task in ["task1", "both"] and self.task1_runner:
            # Replace agent temporarily
            original_agent = self.task1_runner.agent
            self.task1_runner.agent = agent

            try:
                # Use parallel processing for sample evaluation
                responses_list, evaluations_list = (
                    self.task1_runner.processor.process_conversations_parallel(
                        dataset,
                        enable_evaluation=True,
                        n_parallel=self.config.n_parallel,
                    )
                )

                # Extract scores from evaluations
                for evaluations in evaluations_list:
                    for turn_key, turn_eval in evaluations.items():
                        if turn_key.startswith("turn_") and "error" not in turn_eval:
                            # Normalize from 0-10 scale to 0-1 scale
                            all_scores.append(turn_eval["overall_score"] / 10.0)
                            total_evaluations += 1
            finally:
                # Restore original agent
                self.task1_runner.agent = original_agent

        # Evaluate on task2 if specified
        if self.config.task in ["task2", "both"] and self.task2_runner:
            # Replace agent temporarily
            original_agent = self.task2_runner.agent
            self.task2_runner.agent = agent

            try:
                # Use parallel processing for sample evaluation
                responses_list, evaluations_list = (
                    self.task2_runner.processor.process_conversations_parallel(
                        dataset,
                        enable_evaluation=True,
                        n_parallel=self.config.n_parallel,
                    )
                )

                # Extract scores from evaluations
                for evaluations in evaluations_list:
                    for turn_key, turn_eval in evaluations.items():
                        if turn_key.startswith("turn_") and "error" not in turn_eval:
                            # Normalize from 0-10 scale to 0-1 scale
                            all_scores.append(turn_eval["overall_score"] / 10.0)
                            total_evaluations += 1
            finally:
                # Restore original agent
                self.task2_runner.agent = original_agent

        # Calculate overall score
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        evaluation_details = {
            "total_samples": total_evaluations,
            "sample_scores": all_scores,
        }

        return overall_score, evaluation_details

    def _initialize_beam_search(self) -> None:
        """Initialize beam search with the initial prompt."""
        if not self.config.enable_beam_search:
            return

        logger.info(f"🔍 Initializing beam search with width {self.config.beam_width}")

        # Create initial candidate
        initial_candidate = PromptCandidate(
            function_prompt=self.state.current_function_prompt,
            dialogue_prompt=self.state.current_dialogue_prompt,
            score=self.state.best_score,
            evaluation_count=1,
        )

        self.state.beam_candidates = [initial_candidate]

    def _beam_search_step(self, gradients: List, gradient_summary: str) -> None:
        """
        Perform one step of beam search.

        Args:
            gradients: Generated gradients for improvement
            gradient_summary: Summary of gradients
        """
        if not self.config.enable_beam_search or not gradients:
            return

        logger.info(
            f"🔍 Beam search step with {len(self.state.beam_candidates)} candidates"
        )

        new_candidates = []

        # Generate candidates from each beam entry
        for beam_candidate in self.state.beam_candidates:
            # Generate multiple prompt variants
            for prompt_type in self.config.prompt_types:
                current_prompt = (
                    beam_candidate.function_prompt
                    if prompt_type == "function"
                    else beam_candidate.dialogue_prompt
                )

                if current_prompt:
                    candidates = self.prompt_editor.generate_prompt_candidates(
                        current_prompt=current_prompt,
                        gradients=gradients,
                        gradient_summary=gradient_summary,
                        prompt_type=prompt_type,
                        num_candidates=2,  # Generate 2 per beam entry
                        gradient_memory=self.state.gradient_memory
                        if self.config.enable_gradient_memory
                        else None,
                    )

                    # Evaluate candidates and create new beam entries
                    for candidate_prompt in candidates:
                        if prompt_type == "function":
                            new_function_prompt = candidate_prompt
                            new_dialogue_prompt = beam_candidate.dialogue_prompt
                        else:
                            new_function_prompt = beam_candidate.function_prompt
                            new_dialogue_prompt = candidate_prompt

                        # Evaluate the new candidate
                        score, _ = self._evaluate_prompts_sample(
                            new_function_prompt,
                            new_dialogue_prompt,
                            sample_size=self.config.monte_carlo_sample_size,
                        )

                        new_candidate = PromptCandidate(
                            function_prompt=new_function_prompt,
                            dialogue_prompt=new_dialogue_prompt,
                            score=score,
                            evaluation_count=1,
                            generation_metadata={
                                "parent_score": beam_candidate.score,
                                "prompt_type": prompt_type,
                                "iteration": self.state.iteration,
                            },
                        )
                        new_candidates.append(new_candidate)

        # Keep top-k candidates
        all_candidates = self.state.beam_candidates + new_candidates
        all_candidates.sort(key=lambda x: x.score, reverse=True)

        self.state.beam_candidates = all_candidates[: self.config.beam_width]

        # Update current prompts with the best candidate
        best_candidate = self.state.beam_candidates[0]
        self.state.current_function_prompt = best_candidate.function_prompt
        self.state.current_dialogue_prompt = best_candidate.dialogue_prompt

        logger.info(
            f"🏆 Beam search: keeping top {len(self.state.beam_candidates)} candidates"
        )
        logger.info(f"📊 Best candidate score: {best_candidate.score:.3f}")

    def _update_gradient_memory(self, gradients: List, gradient_summary: str) -> None:
        """
        Update gradient memory with current iteration.

        Args:
            gradients: Current gradients
            gradient_summary: Summary of current gradients
        """
        if not self.config.enable_gradient_memory:
            return

        memory_entry = GradientMemory(
            iteration=self.state.iteration,
            gradients=gradients,
            gradient_summary=gradient_summary,
            timestamp=datetime.now().isoformat(),
        )

        self.state.gradient_memory.append(memory_entry)

        # Keep only recent memory
        if len(self.state.gradient_memory) > self.config.gradient_memory_size:
            self.state.gradient_memory = self.state.gradient_memory[
                -self.config.gradient_memory_size :
            ]

        logger.debug(
            f"💭 Updated gradient memory: {len(self.state.gradient_memory)} entries"
        )

    def _save_results(self) -> None:
        """Save optimization results."""
        logger.info("💾 Saving optimization results")

        # Save final prompts
        final_prompts = {
            "best_function_prompt": self.state.best_function_prompt,
            "best_dialogue_prompt": self.state.best_dialogue_prompt,
            "optimization_summary": {
                "initial_score": self.state.iteration_results[0].score
                if self.state.iteration_results
                else 0,
                "final_score": self.state.best_score,
                "total_iterations": self.state.iteration,
                "threshold_reached": self.state.best_score
                >= self.config.score_threshold,
                "total_evaluations": self.state.total_evaluations,
            },
            "iteration_results": [
                {
                    "iteration": r.iteration,
                    "score": r.score,
                    "improvement": r.improvement,
                    "timestamp": r.timestamp,
                }
                for r in self.state.iteration_results
            ],
        }

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save prompts
        prompts_path = output_path.with_suffix(".prompts.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(final_prompts, f, indent=2, ensure_ascii=False)

        logger.success(f"💾 Saved optimized prompts: {prompts_path}")

        # Save detailed results
        results_path = output_path.with_suffix(".results.json")
        detailed_results = {
            "config": {
                "task": self.config.task,
                "prompt_types": self.config.prompt_types,
                "score_threshold": self.config.score_threshold,
                "max_iterations": self.config.max_iterations,
            },
            "state": {
                "best_score": self.state.best_score,
                "total_iterations": self.state.iteration,
                "total_evaluations": self.state.total_evaluations,
                "start_time": self.state.start_time,
            },
            "iteration_results": [
                {
                    "iteration": r.iteration,
                    "score": r.score,
                    "response_score": r.response_score,
                    "functions_score": r.functions_score,
                    "improvement": r.improvement,
                    "total_samples": r.total_samples,
                    "failed_samples": r.failed_samples,
                    "gradient_count": r.gradient_count,
                    "timestamp": r.timestamp,
                }
                for r in self.state.iteration_results
            ],
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        logger.success(f"📊 Saved detailed results: {results_path}")
