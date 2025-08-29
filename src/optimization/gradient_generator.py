"""Gradient generation for prompt optimization using language models."""

import json
import litellm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Gradient:
    """Represents a gradient (feedback) for prompt improvement."""

    query: str
    generated_response: str
    expected_response: str
    generated_functions: List[Dict]
    expected_functions: List[Dict]
    score: float
    gradient_type: str  # "positive" or "negative"
    issue_analysis: str
    suggested_improvements: List[str]
    context: Dict


class GradientGenerator:
    """Generates gradients (feedback) from failed samples to improve prompts."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        score_threshold: float = 0.8,
        positive_threshold: float = 0.5,
        include_positive_gradients: bool = True,
        positive_negative_ratio: float = 0.5,
    ):
        """
        Initialize the GradientGenerator.

        Args:
            model: Language model to use for gradient generation
            temperature: Temperature for generation
            score_threshold: Score threshold below which samples are considered failed
            positive_threshold: Score threshold above which samples are considered successful
            include_positive_gradients: Whether to generate positive gradients from successful samples
            positive_negative_ratio: Ratio of positive to negative gradients (0.3 = 30% positive, 70% negative)
        """
        self.model = model
        self.temperature = temperature
        self.score_threshold = score_threshold
        self.positive_threshold = positive_threshold
        self.include_positive_gradients = include_positive_gradients
        self.positive_negative_ratio = positive_negative_ratio
        logger.info(f"🧠 Initialized GradientGenerator with model: {model}")
        logger.info(
            f"🎯 Negative threshold: {score_threshold}, Positive threshold: {positive_threshold}"
        )
        if include_positive_gradients:
            logger.info(
                f"➕ Positive gradients enabled with ratio: {positive_negative_ratio}"
            )

    def generate_gradients(
        self,
        evaluation_results: List[Dict],
        max_samples: int = 20,
        mini_batch_size: Optional[int] = None,
    ) -> List[Gradient]:
        """
        Generate gradients from evaluation results.

        Args:
            evaluation_results: List of evaluation results from task processing
            max_samples: Maximum number of samples to analyze (split between positive/negative)
            mini_batch_size: If specified, randomly sample this many samples total

        Returns:
            List of Gradient objects with improvement suggestions
        """
        logger.info("🔍 Analyzing samples for gradient generation")

        # Filter samples into positive and negative
        failed_samples = self._filter_failed_samples(evaluation_results)
        successful_samples = []

        if self.include_positive_gradients:
            successful_samples = self._filter_successful_samples(evaluation_results)

        if not failed_samples and not successful_samples:
            logger.warning("⚠️ No samples found for gradient generation")
            return []

        # Calculate sample distribution
        total_samples_available = len(failed_samples) + len(successful_samples)

        if mini_batch_size and total_samples_available > mini_batch_size:
            max_samples = mini_batch_size

        # Calculate how many positive vs negative samples to use
        if self.include_positive_gradients and successful_samples:
            positive_count = min(
                int(max_samples * self.positive_negative_ratio), len(successful_samples)
            )
            negative_count = min(max_samples - positive_count, len(failed_samples))
        else:
            positive_count = 0
            negative_count = min(max_samples, len(failed_samples))

        # Sample the data
        import random

        selected_failed = []
        selected_successful = []

        if negative_count > 0:
            # Ensure failed_samples is a proper list
            if not isinstance(failed_samples, list):
                logger.error(f"❌ failed_samples is not a list: {type(failed_samples)}")
                failed_samples = []

            if len(failed_samples) > 0:
                selected_failed = random.sample(
                    failed_samples, min(negative_count, len(failed_samples))
                )

        if positive_count > 0:
            # Ensure successful_samples is a proper list
            if not isinstance(successful_samples, list):
                logger.error(
                    f"❌ successful_samples is not a list: {type(successful_samples)}"
                )
                successful_samples = []

            if len(successful_samples) > 0:
                selected_successful = random.sample(
                    successful_samples, min(positive_count, len(successful_samples))
                )

        logger.info(
            f"📊 Selected {len(selected_failed)} negative and {len(selected_successful)} positive samples"
        )

        gradients = []

        # Generate negative gradients
        for i, sample in enumerate(selected_failed):
            try:
                gradient = self._generate_single_gradient(
                    sample, gradient_type="negative"
                )
                if gradient:
                    gradients.append(gradient)
                    logger.debug(
                        f"✅ Generated negative gradient {i + 1}/{len(selected_failed)}"
                    )
                else:
                    logger.warning(
                        f"⚠️ Failed to generate negative gradient for sample {i + 1}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Error generating negative gradient for sample {i + 1}: {e}"
                )
                continue

        # Generate positive gradients
        for i, sample in enumerate(selected_successful):
            try:
                gradient = self._generate_single_gradient(
                    sample, gradient_type="positive"
                )
                if gradient:
                    gradients.append(gradient)
                    logger.debug(
                        f"✅ Generated positive gradient {i + 1}/{len(selected_successful)}"
                    )
                else:
                    logger.warning(
                        f"⚠️ Failed to generate positive gradient for sample {i + 1}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Error generating positive gradient for sample {i + 1}: {e}"
                )
                continue

        logger.success(
            f"🎯 Generated {len(gradients)} total gradients ({len([g for g in gradients if g.gradient_type == 'negative'])} negative, {len([g for g in gradients if g.gradient_type == 'positive'])} positive)"
        )
        return gradients

    def _filter_failed_samples(self, evaluation_results: List[Dict]) -> List[Dict]:
        """Filter evaluation results to get failed samples."""
        failed_samples = []

        for conv_idx, conv_results in enumerate(evaluation_results):
            # Ensure conv_results is a dictionary
            if not isinstance(conv_results, dict):
                logger.warning(
                    f"⚠️ Skipping non-dict conversation result at index {conv_idx}: {type(conv_results)}"
                )
                continue

            for turn_key, turn_results in conv_results.items():
                if turn_key.startswith("turn_") and "error" not in turn_results:
                    overall_score = turn_results.get("overall_score", 0.0)
                    if overall_score < self.score_threshold:
                        # Extract sample data for gradient generation
                        sample = {
                            "conversation_index": conv_idx,
                            "turn": turn_key,
                            "score": overall_score,
                            "response_score": turn_results.get("response_score", 0.0),
                            "functions_score": turn_results.get("functions_score", 0.0),
                            "evaluation_results": turn_results,
                        }
                        failed_samples.append(sample)

        # Sort by score (worst first)
        failed_samples.sort(key=lambda x: x["score"])

        logger.info(
            f"📊 Found {len(failed_samples)} failed samples (score < {self.score_threshold})"
        )
        return failed_samples

    def _filter_successful_samples(self, evaluation_results: List[Dict]) -> List[Dict]:
        """Filter evaluation results to get successful samples."""
        successful_samples = []

        for conv_idx, conv_results in enumerate(evaluation_results):
            # Ensure conv_results is a dictionary
            if not isinstance(conv_results, dict):
                logger.warning(
                    f"⚠️ Skipping non-dict conversation result at index {conv_idx}: {type(conv_results)}"
                )
                continue

            for turn_key, turn_results in conv_results.items():
                if turn_key.startswith("turn_") and "error" not in turn_results:
                    overall_score = turn_results.get("overall_score", 0.0)
                    if overall_score >= self.positive_threshold:
                        # Extract sample data for gradient generation
                        sample = {
                            "conversation_index": conv_idx,
                            "turn": turn_key,
                            "score": overall_score,
                            "response_score": turn_results.get("response_score", 0.0),
                            "functions_score": turn_results.get("functions_score", 0.0),
                            "evaluation_results": turn_results,
                        }
                        successful_samples.append(sample)

        # Sort by score (best first)
        successful_samples.sort(key=lambda x: x["score"])

        logger.info(
            f"📊 Found {len(successful_samples)} successful samples (score >= {self.positive_threshold})"
        )
        return successful_samples

    def _generate_single_gradient(
        self, sample: Dict, gradient_type: str
    ) -> Optional[Gradient]:
        """Generate a single gradient from a sample."""

        # Create the gradient generation prompt
        prompt = self._create_gradient_prompt(sample, gradient_type)

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )

            analysis_text = response.choices[0].message.content.strip()
            return self._parse_gradient_response(analysis_text, sample, gradient_type)

        except Exception as e:
            logger.error(f"❌ Error calling LM for gradient generation: {e}")
            return None

    def _create_gradient_prompt(self, sample: Dict, gradient_type: str) -> str:
        """Create a prompt for analyzing a sample (either successful or failed)."""

        if gradient_type == "positive":
            return self._create_positive_gradient_prompt(sample)
        else:
            return self._create_negative_gradient_prompt(sample)

    def _create_negative_gradient_prompt(self, sample: Dict) -> str:
        """Create a prompt for analyzing a failed sample."""

        return f"""
You are an expert prompt engineer analyzing why an AI agent failed to perform well on a task. Your goal is to identify specific issues and suggest improvements to the agent's prompts.

## FAILED SAMPLE ANALYSIS

**Performance Scores:**
- Overall Score: {sample["score"]:.2f} (threshold: {self.score_threshold})
- Response Score: {sample["response_score"]:.2f}
- Functions Score: {sample["functions_score"]:.2f}

**Sample Details:**
- Conversation: {sample["conversation_index"]}
- Turn: {sample["turn"]}

## YOUR ANALYSIS TASK

Please analyze this failed sample and provide:

1. **Root Cause Analysis**: What specific issues caused the poor performance?
   - Was it incorrect function calling?
   - Poor response quality?
   - Misunderstanding of context?
   - Inappropriate tone or character behavior?

2. **Prompt Improvement Suggestions**: Provide 3-5 specific, actionable suggestions for improving the prompts:
   - For function calling prompt improvements
   - For dialogue generation prompt improvements
   - For both if applicable

3. **Priority Level**: Rate each suggestion as HIGH, MEDIUM, or LOW priority

## RESPONSE FORMAT

Please respond in this JSON format:

```json
{{
    "root_cause_analysis": "Detailed analysis of what went wrong...",
    "function_prompt_issues": ["Issue 1", "Issue 2", ...],
    "dialogue_prompt_issues": ["Issue 1", "Issue 2", ...],
    "improvement_suggestions": [
        {{
            "type": "function|dialogue|both",
            "priority": "HIGH|MEDIUM|LOW",
            "suggestion": "Specific improvement suggestion...",
            "rationale": "Why this improvement would help..."
        }}
    ],
    "key_themes": ["Theme 1", "Theme 2", ...]
}}
```

Be specific and actionable in your suggestions. Focus on concrete changes that could be made to prompts to address the identified issues.
"""

    def _create_positive_gradient_prompt(self, sample: Dict) -> str:
        """Create a prompt for analyzing a successful sample."""

        return f"""
You are an expert prompt engineer analyzing why an AI agent performed exceptionally well on a task. Your goal is to identify what worked well and suggest how to reinforce these successful patterns in the agent's prompts.

## SUCCESSFUL SAMPLE ANALYSIS

**Performance Scores:**
- Overall Score: {sample["score"]:.2f} (threshold: {self.positive_threshold})
- Response Score: {sample["response_score"]:.2f}
- Functions Score: {sample["functions_score"]:.2f}

**Sample Details:**
- Conversation: {sample["conversation_index"]}
- Turn: {sample["turn"]}

## YOUR ANALYSIS TASK

Please analyze this successful sample and provide:

1. **Success Factor Analysis**: What specific elements contributed to the excellent performance?
   - Was it accurate function calling?
   - High-quality response generation?
   - Good understanding of context?
   - Appropriate tone or character behavior?

2. **Prompt Reinforcement Suggestions**: Provide 3-5 specific suggestions for reinforcing successful patterns:
   - For function calling prompt enhancements
   - For dialogue generation prompt enhancements
   - For both if applicable

3. **Priority Level**: Rate each suggestion as HIGH, MEDIUM, or LOW priority

## RESPONSE FORMAT

Please respond in this JSON format:

```json
{{
    "root_cause_analysis": "Detailed analysis of what worked well...",
    "function_prompt_issues": ["Strength 1", "Strength 2", ...],
    "dialogue_prompt_issues": ["Strength 1", "Strength 2", ...],
    "improvement_suggestions": [
        {{
            "type": "function|dialogue|both",
            "priority": "HIGH|MEDIUM|LOW",
            "suggestion": "Specific reinforcement suggestion...",
            "rationale": "Why reinforcing this pattern would help..."
        }}
    ],
    "key_themes": ["Theme 1", "Theme 2", ...]
}}
```

Focus on identifying patterns that can be reinforced or extended to improve overall performance. Think about what made this sample successful and how to encourage similar behavior.
"""

    def _parse_gradient_response(
        self, response_text: str, sample: Dict, gradient_type: str
    ) -> Optional[Gradient]:
        """Parse the LM response into a Gradient object."""
        try:
            # Extract JSON from the response
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 7)

            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start + 7 : json_end].strip()
            else:
                # Try to parse the entire response as JSON
                json_text = response_text.strip()

            analysis = json.loads(json_text)

            # Create suggestions list from the structured response
            suggestions = []
            for suggestion in analysis.get("improvement_suggestions", []):
                suggestions.append(
                    f"[{suggestion.get('priority', 'MEDIUM')}] {suggestion.get('suggestion', '')} - {suggestion.get('rationale', '')}"
                )

            return Gradient(
                query=f"Conversation {sample['conversation_index']}, {sample['turn']}",
                generated_response="",  # Would need actual data
                expected_response="",  # Would need actual data
                generated_functions=[],  # Would need actual data
                expected_functions=[],  # Would need actual data
                score=sample["score"],
                gradient_type=gradient_type,
                issue_analysis=analysis.get("root_cause_analysis", ""),
                suggested_improvements=suggestions,
                context={
                    "sample": sample,
                    "function_issues": analysis.get("function_prompt_issues", []),
                    "dialogue_issues": analysis.get("dialogue_prompt_issues", []),
                    "themes": analysis.get("key_themes", []),
                },
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"❌ Error parsing gradient response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None

    def summarize_gradients(self, gradients: List[Gradient]) -> Dict[str, any]:
        """Summarize gradients to identify common patterns and themes."""
        if not gradients:
            return {}

        # Separate positive and negative gradients
        positive_gradients = [g for g in gradients if g.gradient_type == "positive"]
        negative_gradients = [g for g in gradients if g.gradient_type == "negative"]

        # Collect themes and issues for each type
        negative_themes = []
        negative_function_issues = []
        negative_dialogue_issues = []
        negative_high_priority = []

        positive_themes = []
        positive_function_strengths = []
        positive_dialogue_strengths = []
        positive_high_priority = []

        # Process negative gradients
        for gradient in negative_gradients:
            negative_themes.extend(gradient.context.get("themes", []))
            negative_function_issues.extend(gradient.context.get("function_issues", []))
            negative_dialogue_issues.extend(gradient.context.get("dialogue_issues", []))

            # Extract high priority suggestions
            for suggestion in gradient.suggested_improvements:
                if suggestion.startswith("[HIGH]"):
                    negative_high_priority.append(suggestion)

        # Process positive gradients
        for gradient in positive_gradients:
            positive_themes.extend(gradient.context.get("themes", []))
            positive_function_strengths.extend(
                gradient.context.get("function_issues", [])
            )  # These are strengths for positive
            positive_dialogue_strengths.extend(
                gradient.context.get("dialogue_issues", [])
            )  # These are strengths for positive

            # Extract high priority suggestions
            for suggestion in gradient.suggested_improvements:
                if suggestion.startswith("[HIGH]"):
                    positive_high_priority.append(suggestion)

        # Count occurrences
        negative_theme_counts = {}
        for theme in negative_themes:
            negative_theme_counts[theme] = negative_theme_counts.get(theme, 0) + 1

        positive_theme_counts = {}
        for theme in positive_themes:
            positive_theme_counts[theme] = positive_theme_counts.get(theme, 0) + 1

        summary = {
            "total_gradients": len(gradients),
            "positive_gradients": len(positive_gradients),
            "negative_gradients": len(negative_gradients),
            "average_score": sum(g.score for g in gradients) / len(gradients),
        }

        # Add negative gradient analysis
        if negative_gradients:
            summary["negative_analysis"] = {
                "average_score": sum(g.score for g in negative_gradients)
                / len(negative_gradients),
                "common_themes": sorted(
                    negative_theme_counts.items(), key=lambda x: x[1], reverse=True
                )[:5],
                "function_issues": list(set(negative_function_issues)),
                "dialogue_issues": list(set(negative_dialogue_issues)),
                "high_priority_suggestions": negative_high_priority[:10],
            }

        # Add positive gradient analysis
        if positive_gradients:
            summary["positive_analysis"] = {
                "average_score": sum(g.score for g in positive_gradients)
                / len(positive_gradients),
                "common_themes": sorted(
                    positive_theme_counts.items(), key=lambda x: x[1], reverse=True
                )[:5],
                "function_strengths": list(set(positive_function_strengths)),
                "dialogue_strengths": list(set(positive_dialogue_strengths)),
                "high_priority_reinforcements": positive_high_priority[:10],
            }

        return summary
