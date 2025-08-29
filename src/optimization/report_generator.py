"""Report generation for prompt optimization results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from .config import OptimizationState, OptimizationConfig, IterationResult


class ReportGenerator:
    """Generates comprehensive reports for prompt optimization results."""

    def __init__(self):
        """Initialize the ReportGenerator."""
        logger.info("📊 Initialized ReportGenerator")

    def generate_final_report(
        self, state: OptimizationState, config: OptimizationConfig, output_path: str
    ) -> str:
        """
        Generate a comprehensive final optimization report.

        Args:
            state: Final optimization state
            config: Optimization configuration
            output_path: Path to save the report

        Returns:
            Path to the generated report
        """
        logger.info("📊 Generating final optimization report")

        # Prepare report data
        report_data = {
            "metadata": self._generate_metadata(config),
            "summary": self._generate_summary(state, config),
            "optimization_progress": self._generate_progress_analysis(state),
            "final_prompts": self._generate_final_prompts(state),
            "performance_analysis": self._generate_performance_analysis(state),
            "iteration_details": self._generate_iteration_details(state),
            "recommendations": self._generate_recommendations(state, config),
        }

        # Save JSON report
        json_path = Path(output_path).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Generate human-readable report
        markdown_path = Path(output_path).with_suffix(".md")
        self._generate_markdown_report(report_data, markdown_path)

        logger.success(f"📊 Generated optimization report: {json_path}")
        logger.success(f"📄 Generated markdown report: {markdown_path}")

        return str(json_path)

    def generate_iteration_summary(
        self, iteration_result: IterationResult, previous_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate a summary for a single iteration.

        Args:
            iteration_result: Results from the iteration
            previous_score: Score from the previous iteration

        Returns:
            Summary dictionary
        """
        improvement = iteration_result.score - previous_score

        return {
            "iteration": iteration_result.iteration,
            "timestamp": iteration_result.timestamp,
            "performance": {
                "overall_score": iteration_result.score,
                "response_score": iteration_result.response_score,
                "functions_score": iteration_result.functions_score,
                "improvement": improvement,
                "improvement_percent": (improvement / previous_score * 100)
                if previous_score > 0
                else 0,
            },
            "evaluation_stats": {
                "total_samples": iteration_result.total_samples,
                "failed_samples": iteration_result.failed_samples,
                "success_rate": (
                    (iteration_result.total_samples - iteration_result.failed_samples)
                    / iteration_result.total_samples
                    * 100
                )
                if iteration_result.total_samples > 0
                else 0,
            },
            "gradient_count": iteration_result.gradient_count,
        }

    def _generate_metadata(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            "generated_at": datetime.now().isoformat(),
            "task": config.task,
            "data_path": config.data_path,
            "prompt_types": config.prompt_types,
            "optimization_config": {
                "score_threshold": config.score_threshold,
                "max_iterations": config.max_iterations,
                "min_improvement_threshold": config.min_improvement_threshold,
                "sample_size": config.sample_size,
            },
            "model_config": {
                "gradient_model": config.gradient_model,
                "editor_model": config.editor_model,
                "evaluator_model": config.evaluator_model,
            },
        }

    def _generate_summary(
        self, state: OptimizationState, config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Generate optimization summary."""
        total_improvement = state.best_score - (
            state.iteration_results[0].score if state.iteration_results else 0
        )

        return {
            "total_iterations": state.iteration,
            "total_evaluations": state.total_evaluations,
            "optimization_completed": state.best_score >= config.score_threshold,
            "threshold_reached": state.best_score >= config.score_threshold,
            "performance": {
                "initial_score": state.iteration_results[0].score
                if state.iteration_results
                else 0,
                "final_score": state.best_score,
                "total_improvement": total_improvement,
                "improvement_percent": (
                    total_improvement / state.iteration_results[0].score * 100
                )
                if state.iteration_results and state.iteration_results[0].score > 0
                else 0,
            },
            "optimization_time": self._calculate_optimization_time(state),
        }

    def _generate_progress_analysis(self, state: OptimizationState) -> Dict[str, Any]:
        """Analyze optimization progress over iterations."""
        if not state.iteration_results:
            return {}

        scores = [result.score for result in state.iteration_results]
        improvements = [result.improvement for result in state.iteration_results]

        return {
            "score_progression": scores,
            "improvement_progression": improvements,
            "best_iteration": max(range(len(scores)), key=lambda i: scores[i]),
            "convergence_analysis": {
                "converged": self._check_convergence(scores),
                "plateau_detection": self._detect_plateau(scores),
                "instability_warning": self._check_instability(scores),
            },
        }

    def _generate_final_prompts(self, state: OptimizationState) -> Dict[str, Any]:
        """Generate final prompt information."""
        return {
            "best_function_prompt": state.best_function_prompt,
            "best_dialogue_prompt": state.best_dialogue_prompt,
            "current_function_prompt": state.current_function_prompt,
            "current_dialogue_prompt": state.current_dialogue_prompt,
            "prompt_evolution": self._analyze_prompt_evolution(state),
        }

    def _generate_performance_analysis(
        self, state: OptimizationState
    ) -> Dict[str, Any]:
        """Analyze performance metrics across iterations."""
        if not state.iteration_results:
            return {}

        response_scores = [r.response_score for r in state.iteration_results]
        function_scores = [r.functions_score for r in state.iteration_results]

        return {
            "response_performance": {
                "initial": response_scores[0],
                "final": response_scores[-1],
                "best": max(response_scores),
                "improvement": response_scores[-1] - response_scores[0],
            },
            "function_performance": {
                "initial": function_scores[0],
                "final": function_scores[-1],
                "best": max(function_scores),
                "improvement": function_scores[-1] - function_scores[0],
            },
            "evaluation_statistics": {
                "total_samples_processed": sum(
                    r.total_samples for r in state.iteration_results
                ),
                "total_failed_samples": sum(
                    r.failed_samples for r in state.iteration_results
                ),
                "average_success_rate": self._calculate_average_success_rate(
                    state.iteration_results
                ),
            },
        }

    def _generate_iteration_details(
        self, state: OptimizationState
    ) -> List[Dict[str, Any]]:
        """Generate detailed information for each iteration."""
        details = []
        previous_score = 0.0

        for result in state.iteration_results:
            details.append(self.generate_iteration_summary(result, previous_score))
            previous_score = result.score

        return details

    def _generate_recommendations(
        self, state: OptimizationState, config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Generate recommendations based on optimization results."""
        recommendations = []

        # Check if threshold was reached
        if state.best_score < config.score_threshold:
            recommendations.append(
                {
                    "type": "threshold_not_reached",
                    "message": f"Score threshold {config.score_threshold} was not reached (best: {state.best_score:.3f})",
                    "suggestions": [
                        "Consider increasing max_iterations",
                        "Try different gradient or editor models",
                        "Adjust the score threshold if current performance is acceptable",
                        "Analyze failed samples for additional insights",
                    ],
                }
            )

        # Check for convergence issues
        if len(state.iteration_results) >= 3:
            recent_improvements = [r.improvement for r in state.iteration_results[-3:]]
            if all(
                imp < config.min_improvement_threshold for imp in recent_improvements
            ):
                recommendations.append(
                    {
                        "type": "convergence_plateau",
                        "message": "Optimization appears to have plateaued",
                        "suggestions": [
                            "Try a different prompt edit strategy",
                            "Increase gradient model temperature for more diverse feedback",
                            "Consider analyzing different types of failed samples",
                        ],
                    }
                )

        # Performance-specific recommendations
        if state.iteration_results:
            latest = state.iteration_results[-1]
            if latest.response_score < latest.functions_score:
                recommendations.append(
                    {
                        "type": "response_improvement_needed",
                        "message": "Response quality is lagging behind function calling performance",
                        "suggestions": [
                            "Focus optimization on dialogue prompts",
                            "Analyze response quality issues in failed samples",
                            "Consider using a more powerful model for dialogue generation",
                        ],
                    }
                )
            elif latest.functions_score < latest.response_score:
                recommendations.append(
                    {
                        "type": "function_improvement_needed",
                        "message": "Function calling performance is lagging behind response quality",
                        "suggestions": [
                            "Focus optimization on function calling prompts",
                            "Analyze function selection and parameter issues",
                            "Consider improving function calling examples and guidelines",
                        ],
                    }
                )

        return {
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(state, config),
        }

    def _generate_next_steps(
        self, state: OptimizationState, config: OptimizationConfig
    ) -> List[str]:
        """Generate suggested next steps based on results."""
        next_steps = []

        if state.best_score >= config.score_threshold:
            next_steps.extend(
                [
                    "Deploy the optimized prompts to production",
                    "Monitor performance on new data",
                    "Consider fine-tuning the prompts further on domain-specific data",
                ]
            )
        else:
            next_steps.extend(
                [
                    "Analyze remaining failure cases in detail",
                    "Consider collecting more training data",
                    "Experiment with different optimization strategies",
                ]
            )

        next_steps.append("Set up continuous monitoring of prompt performance")
        next_steps.append("Document the optimized prompts and optimization process")

        return next_steps

    def _calculate_optimization_time(self, state: OptimizationState) -> Optional[str]:
        """Calculate total optimization time."""
        if not state.start_time or not state.iteration_results:
            return None

        try:
            start_time = datetime.fromisoformat(state.start_time)
            end_time = datetime.fromisoformat(state.iteration_results[-1].timestamp)
            duration = end_time - start_time

            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            elif minutes > 0:
                return f"{int(minutes)}m {int(seconds)}s"
            else:
                return f"{int(seconds)}s"
        except:
            return None

    def _check_convergence(self, scores: List[float], window: int = 3) -> bool:
        """Check if optimization has converged."""
        if len(scores) < window:
            return False

        recent_scores = scores[-window:]
        variance = sum(
            (score - sum(recent_scores) / len(recent_scores)) ** 2
            for score in recent_scores
        ) / len(recent_scores)
        return variance < 0.001  # Low variance indicates convergence

    def _detect_plateau(self, scores: List[float], window: int = 3) -> bool:
        """Detect if optimization has reached a plateau."""
        if len(scores) < window:
            return False

        recent_scores = scores[-window:]
        max_change = max(recent_scores) - min(recent_scores)
        return max_change < 0.01  # Small change indicates plateau

    def _check_instability(self, scores: List[float], window: int = 5) -> bool:
        """Check for optimization instability."""
        if len(scores) < window:
            return False

        recent_scores = scores[-window:]
        ups = sum(
            1
            for i in range(1, len(recent_scores))
            if recent_scores[i] > recent_scores[i - 1]
        )
        downs = len(recent_scores) - 1 - ups

        # High oscillation indicates instability
        return min(ups, downs) >= 2

    def _analyze_prompt_evolution(self, state: OptimizationState) -> Dict[str, Any]:
        """Analyze how prompts evolved during optimization."""
        function_prompt_changes = 0
        dialogue_prompt_changes = 0

        prev_function = (
            state.iteration_results[0].function_prompt
            if state.iteration_results
            else None
        )
        prev_dialogue = (
            state.iteration_results[0].dialogue_prompt
            if state.iteration_results
            else None
        )

        for result in state.iteration_results[1:]:
            if result.function_prompt != prev_function:
                function_prompt_changes += 1
                prev_function = result.function_prompt

            if result.dialogue_prompt != prev_dialogue:
                dialogue_prompt_changes += 1
                prev_dialogue = result.dialogue_prompt

        return {
            "function_prompt_changes": function_prompt_changes,
            "dialogue_prompt_changes": dialogue_prompt_changes,
            "total_prompt_updates": function_prompt_changes + dialogue_prompt_changes,
        }

    def _calculate_average_success_rate(self, results: List[IterationResult]) -> float:
        """Calculate average success rate across iterations."""
        if not results:
            return 0.0

        total_samples = sum(r.total_samples for r in results)
        total_failed = sum(r.failed_samples for r in results)

        if total_samples == 0:
            return 0.0

        return (total_samples - total_failed) / total_samples * 100

    def _generate_markdown_report(
        self, report_data: Dict[str, Any], output_path: Path
    ) -> None:
        """Generate a human-readable markdown report."""
        metadata = report_data["metadata"]
        summary = report_data["summary"]

        markdown_content = f"""# Prompt Optimization Report

Generated on: {metadata["generated_at"]}

## Summary

- **Task**: {metadata["task"]}
- **Total Iterations**: {summary["total_iterations"]}
- **Total Evaluations**: {summary["total_evaluations"]}
- **Threshold Reached**: {"✅ Yes" if summary["threshold_reached"] else "❌ No"}
- **Optimization Time**: {summary.get("optimization_time", "Unknown")}

### Performance
- **Initial Score**: {summary["performance"]["initial_score"]:.3f}
- **Final Score**: {summary["performance"]["final_score"]:.3f}
- **Total Improvement**: {summary["performance"]["total_improvement"]:.3f} ({summary["performance"]["improvement_percent"]:.1f}%)

## Optimization Progress

"""

        # Add progress chart data
        progress = report_data["optimization_progress"]
        if progress.get("score_progression"):
            markdown_content += "### Score Progression\n\n"
            for i, score in enumerate(progress["score_progression"]):
                markdown_content += f"- Iteration {i}: {score:.3f}\n"
            markdown_content += "\n"

        # Add recommendations
        recommendations = report_data.get("recommendations", {})
        if recommendations.get("recommendations"):
            markdown_content += "## Recommendations\n\n"
            for rec in recommendations["recommendations"]:
                markdown_content += f"### {rec['type'].replace('_', ' ').title()}\n\n"
                markdown_content += f"{rec['message']}\n\n"
                if rec.get("suggestions"):
                    markdown_content += "**Suggestions:**\n"
                    for suggestion in rec["suggestions"]:
                        markdown_content += f"- {suggestion}\n"
                    markdown_content += "\n"

        # Add next steps
        if recommendations.get("next_steps"):
            markdown_content += "## Next Steps\n\n"
            for step in recommendations["next_steps"]:
                markdown_content += f"- {step}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"📄 Generated markdown report: {output_path}")
