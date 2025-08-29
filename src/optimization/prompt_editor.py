"""Prompt editing using language models based on gradient feedback."""

import json
import litellm
from typing import List, Dict, Optional, Literal, Tuple
from loguru import logger

from .gradient_generator import Gradient
from .config import GradientMemory


class PromptEditor:
    """Edits prompts based on gradient feedback using language models."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        edit_strategy: Literal["incremental", "replacement"] = "incremental",
    ):
        """
        Initialize the PromptEditor.

        Args:
            model: Language model to use for prompt editing
            temperature: Temperature for generation (lower for more focused edits)
            edit_strategy: Strategy for editing - "incremental" or "replacement"
        """
        self.model = model
        self.temperature = temperature
        self.edit_strategy = edit_strategy
        logger.info(
            f"✏️ Initialized PromptEditor with model: {model}, strategy: {edit_strategy}"
        )

    def edit_function_prompt(
        self, current_prompt: str, gradients: List[Gradient], gradient_summary: Dict
    ) -> str:
        """
        Edit the function calling prompt based on gradients.

        Args:
            current_prompt: Current function calling prompt
            gradients: List of gradients with improvement suggestions
            gradient_summary: Summary of common issues and themes

        Returns:
            Improved function calling prompt
        """
        logger.info("✏️ Editing function calling prompt based on gradients")

        # Extract function-specific issues and suggestions
        function_issues = gradient_summary.get("function_issues", [])
        function_suggestions = []

        for gradient in gradients:
            for suggestion in gradient.suggested_improvements:
                if "function" in suggestion.lower():
                    function_suggestions.append(suggestion)

        if not function_issues and not function_suggestions:
            logger.warning("⚠️ No function-related feedback found in gradients")
            return current_prompt

        # Create editing prompt
        editing_prompt = self._create_function_editing_prompt(
            current_prompt, function_issues, function_suggestions, gradient_summary
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": editing_prompt}],
                temperature=self.temperature,
            )

            edited_prompt = self._extract_edited_prompt(
                response.choices[0].message.content
            )

            if edited_prompt:
                logger.success("✅ Successfully edited function calling prompt")
                return edited_prompt
            else:
                logger.warning("⚠️ Failed to extract edited prompt, returning original")
                return current_prompt

        except Exception as e:
            logger.error(f"❌ Error editing function prompt: {e}")
            return current_prompt

    def edit_dialogue_prompt(
        self, current_prompt: str, gradients: List[Gradient], gradient_summary: Dict
    ) -> str:
        """
        Edit the dialogue generation prompt based on gradients.

        Args:
            current_prompt: Current dialogue generation prompt
            gradients: List of gradients with improvement suggestions
            gradient_summary: Summary of common issues and themes

        Returns:
            Improved dialogue generation prompt
        """
        logger.info("✏️ Editing dialogue generation prompt based on gradients")

        # Extract dialogue-specific issues and suggestions
        dialogue_issues = gradient_summary.get("dialogue_issues", [])
        dialogue_suggestions = []

        for gradient in gradients:
            for suggestion in gradient.suggested_improvements:
                if "dialogue" in suggestion.lower() or "response" in suggestion.lower():
                    dialogue_suggestions.append(suggestion)

        if not dialogue_issues and not dialogue_suggestions:
            logger.warning("⚠️ No dialogue-related feedback found in gradients")
            return current_prompt

        # Create editing prompt
        editing_prompt = self._create_dialogue_editing_prompt(
            current_prompt, dialogue_issues, dialogue_suggestions, gradient_summary
        )

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": editing_prompt}],
                temperature=self.temperature,
            )

            edited_prompt = self._extract_edited_prompt(
                response.choices[0].message.content
            )

            if edited_prompt:
                logger.success("✅ Successfully edited dialogue generation prompt")
                return edited_prompt
            else:
                logger.warning("⚠️ Failed to extract edited prompt, returning original")
                return current_prompt

        except Exception as e:
            logger.error(f"❌ Error editing dialogue prompt: {e}")
            return current_prompt

    def generate_prompt_candidates(
        self,
        current_prompt: str,
        gradients: List[Gradient],
        gradient_summary: Dict,
        prompt_type: str,
        num_candidates: int = 3,
        gradient_memory: Optional[List[GradientMemory]] = None,
    ) -> List[str]:
        """
        Generate multiple prompt candidates for selection.

        Args:
            current_prompt: Current prompt to improve
            gradients: List of gradients with improvement suggestions
            gradient_summary: Summary of common issues and themes
            prompt_type: Type of prompt ("function" or "dialogue")
            num_candidates: Number of candidates to generate
            gradient_memory: Previous gradient history for context

        Returns:
            List of candidate prompts
        """
        logger.info(
            f"🔄 Generating {num_candidates} candidates for {prompt_type} prompt"
        )

        candidates = []

        # Create memory context if available
        memory_context = (
            self._create_memory_context(gradient_memory) if gradient_memory else ""
        )

        for i in range(num_candidates):
            try:
                if prompt_type == "function":
                    candidate = self._generate_function_candidate(
                        current_prompt, gradients, gradient_summary, i, memory_context
                    )
                else:
                    candidate = self._generate_dialogue_candidate(
                        current_prompt, gradients, gradient_summary, i, memory_context
                    )

                if candidate and candidate != current_prompt:
                    candidates.append(candidate)
                    logger.debug(f"✅ Generated candidate {i + 1}/{num_candidates}")
                else:
                    logger.warning(
                        f"⚠️ Candidate {i + 1} was identical to current prompt"
                    )

            except Exception as e:
                logger.error(f"❌ Error generating candidate {i + 1}: {e}")
                continue

        logger.success(f"🎯 Generated {len(candidates)} unique candidates")
        return candidates

    def _create_memory_context(self, gradient_memory: List[GradientMemory]) -> str:
        """Create context from gradient memory."""
        if not gradient_memory:
            return ""

        memory_text = "\n## GRADIENT HISTORY (Previous Iterations)\n"
        for memory in gradient_memory[-3:]:  # Use last 3 iterations
            memory_text += f"\n**Iteration {memory.iteration}:**\n"
            memory_text += f"- {memory.gradient_summary}\n"
            if memory.gradients:
                memory_text += f"- Number of gradients: {len(memory.gradients)}\n"

        return memory_text

    def _generate_function_candidate(
        self,
        current_prompt: str,
        gradients: List[Gradient],
        gradient_summary: Dict,
        candidate_index: int,
        memory_context: str,
    ) -> Optional[str]:
        """Generate a function prompt candidate with variation."""

        # Create different approaches for each candidate
        approaches = [
            "Focus on clarity and specificity",
            "Emphasize examples and edge cases",
            "Prioritize structure and organization",
        ]

        approach = approaches[candidate_index % len(approaches)]

        # Extract function-specific issues and suggestions
        function_issues = gradient_summary.get("function_issues", [])
        function_suggestions = []

        for gradient in gradients:
            for suggestion in gradient.suggested_improvements:
                if "function" in suggestion.lower():
                    function_suggestions.append(suggestion)

        editing_prompt = f"""
You are an expert prompt engineer creating a variant of a function calling prompt. 

**APPROACH FOR THIS CANDIDATE: {approach}**

{memory_context}

## CURRENT FUNCTION CALLING PROMPT
```
{current_prompt}
```

## IDENTIFIED ISSUES
{chr(10).join([f"- {issue}" for issue in function_issues]) if function_issues else "No specific issues identified."}

## IMPROVEMENT SUGGESTIONS
{chr(10).join([f"- {suggestion}" for suggestion in function_suggestions]) if function_suggestions else "No specific suggestions provided."}

## YOUR TASK
Create an improved version with the specified approach. Generate a distinctly different variant that addresses the issues while following the approach guidance.

**RESPONSE FORMAT:**
Provide your improved prompt within triple backticks:

```
[Your improved function calling prompt here]
```
"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": editing_prompt}],
                temperature=self.temperature + (candidate_index * 0.1),  # Add variation
            )

            return self._extract_edited_prompt(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"❌ Error generating function candidate: {e}")
            return None

    def _generate_dialogue_candidate(
        self,
        current_prompt: str,
        gradients: List[Gradient],
        gradient_summary: Dict,
        candidate_index: int,
        memory_context: str,
    ) -> Optional[str]:
        """Generate a dialogue prompt candidate with variation."""

        # Create different approaches for each candidate
        approaches = [
            "Focus on natural conversation flow and persona consistency",
            "Emphasize context awareness and response relevance",
            "Prioritize clarity and user engagement",
        ]

        approach = approaches[candidate_index % len(approaches)]

        # Extract dialogue-specific issues and suggestions
        dialogue_issues = gradient_summary.get("dialogue_issues", [])
        dialogue_suggestions = []

        for gradient in gradients:
            for suggestion in gradient.suggested_improvements:
                if "dialogue" in suggestion.lower() or "response" in suggestion.lower():
                    dialogue_suggestions.append(suggestion)

        editing_prompt = f"""
You are an expert prompt engineer creating a variant of a dialogue generation prompt.

**APPROACH FOR THIS CANDIDATE: {approach}**

{memory_context}

## CURRENT DIALOGUE GENERATION PROMPT
```
{current_prompt}
```

## IDENTIFIED ISSUES
{chr(10).join([f"- {issue}" for issue in dialogue_issues]) if dialogue_issues else "No specific issues identified."}

## IMPROVEMENT SUGGESTIONS
{chr(10).join([f"- {suggestion}" for suggestion in dialogue_suggestions]) if dialogue_suggestions else "No specific suggestions provided."}

## YOUR TASK
Create an improved version with the specified approach. Generate a distinctly different variant that addresses the issues while following the approach guidance.

**RESPONSE FORMAT:**
Provide your improved prompt within triple backticks:

```
[Your improved dialogue generation prompt here]
```
"""

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": editing_prompt}],
                temperature=self.temperature + (candidate_index * 0.1),  # Add variation
            )

            return self._extract_edited_prompt(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"❌ Error generating dialogue candidate: {e}")
            return None

    def _create_function_editing_prompt(
        self,
        current_prompt: str,
        issues: List[str],
        suggestions: List[str],
        summary: Dict,
    ) -> str:
        """Create a prompt for editing the function calling prompt."""

        strategy_instruction = ""
        if self.edit_strategy == "incremental":
            strategy_instruction = """
**EDITING STRATEGY: INCREMENTAL IMPROVEMENT**
Make targeted improvements to address the identified issues while preserving the overall structure and effective parts of the current prompt. Focus on:
- Adding clarifications where needed
- Improving specific sections that are causing issues
- Enhancing examples or guidelines
- Maintaining what's already working well
"""
        else:
            strategy_instruction = """
**EDITING STRATEGY: FULL REPLACEMENT**
Create a new prompt that addresses all identified issues while maintaining the core functionality. You have more freedom to restructure and rewrite sections as needed.
"""

        return f"""
You are an expert prompt engineer tasked with improving a function calling prompt for an AI agent based on performance feedback.

{strategy_instruction}

## CURRENT FUNCTION CALLING PROMPT
```
{current_prompt}
```

## IDENTIFIED ISSUES
{chr(10).join([f"- {issue}" for issue in issues]) if issues else "No specific issues identified."}

## IMPROVEMENT SUGGESTIONS
{chr(10).join([f"- {suggestion}" for suggestion in suggestions]) if suggestions else "No specific suggestions provided."}

## PERFORMANCE CONTEXT
- Total failed samples analyzed: {summary.get("total_gradients", 0)}
- Average score of failed samples: {summary.get("average_score", 0):.2f}
- Common themes: {", ".join([theme[0] for theme in summary.get("common_themes", [])[:3]])}

## YOUR TASK
Improve the function calling prompt to address the identified issues and suggestions. Focus on:

1. **Clarity**: Make instructions clearer and more specific
2. **Coverage**: Ensure all important scenarios are covered
3. **Examples**: Add or improve examples where helpful
4. **Structure**: Organize information logically
5. **Robustness**: Handle edge cases and ambiguous situations

## RESPONSE FORMAT
Provide your improved prompt within triple backticks:

```
[Your improved function calling prompt here]
```

Focus on making concrete improvements that directly address the identified issues while maintaining the prompt's effectiveness.
"""

    def _create_dialogue_editing_prompt(
        self,
        current_prompt: str,
        issues: List[str],
        suggestions: List[str],
        summary: Dict,
    ) -> str:
        """Create a prompt for editing the dialogue generation prompt."""

        strategy_instruction = ""
        if self.edit_strategy == "incremental":
            strategy_instruction = """
**EDITING STRATEGY: INCREMENTAL IMPROVEMENT**
Make targeted improvements to address the identified issues while preserving the overall structure and effective parts of the current prompt. Focus on:
- Adding clarifications where needed
- Improving specific sections that are causing issues
- Enhancing character guidelines or examples
- Maintaining what's already working well
"""
        else:
            strategy_instruction = """
**EDITING STRATEGY: FULL REPLACEMENT**
Create a new prompt that addresses all identified issues while maintaining the core functionality. You have more freedom to restructure and rewrite sections as needed.
"""

        return f"""
You are an expert prompt engineer tasked with improving a dialogue generation prompt for an AI agent based on performance feedback.

{strategy_instruction}

## CURRENT DIALOGUE GENERATION PROMPT
```
{current_prompt}
```

## IDENTIFIED ISSUES
{chr(10).join([f"- {issue}" for issue in issues]) if issues else "No specific issues identified."}

## IMPROVEMENT SUGGESTIONS
{chr(10).join([f"- {suggestion}" for suggestion in suggestions]) if suggestions else "No specific suggestions provided."}

## PERFORMANCE CONTEXT
- Total failed samples analyzed: {summary.get("total_gradients", 0)}
- Average score of failed samples: {summary.get("average_score", 0):.2f}
- Common themes: {", ".join([theme[0] for theme in summary.get("common_themes", [])[:3]])}

## YOUR TASK
Improve the dialogue generation prompt to address the identified issues and suggestions. Focus on:

1. **Character Consistency**: Ensure the agent stays in character
2. **Response Quality**: Improve response relevance and helpfulness
3. **Tone and Style**: Match appropriate conversational style
4. **Context Awareness**: Better use of provided context and knowledge
5. **Natural Flow**: Make dialogue feel more natural and engaging

## RESPONSE FORMAT
Provide your improved prompt within triple backticks:

```
[Your improved dialogue generation prompt here]
```

Focus on making concrete improvements that directly address the identified issues while maintaining the prompt's effectiveness for character roleplay.
"""

    def _extract_edited_prompt(self, response_text: str) -> Optional[str]:
        """Extract the edited prompt from the LM response."""
        try:
            # Look for content within triple backticks
            start_marker = "```"
            start_idx = response_text.find(start_marker)

            if start_idx == -1:
                logger.warning("⚠️ No backticks found in response")
                return None

            # Skip the opening backticks and any language identifier
            content_start = start_idx + 3
            next_newline = response_text.find("\n", content_start)
            if next_newline != -1:
                content_start = next_newline + 1

            # Find closing backticks
            end_idx = response_text.find("```", content_start)
            if end_idx == -1:
                logger.warning("⚠️ No closing backticks found in response")
                return None

            edited_prompt = response_text[content_start:end_idx].strip()

            if len(edited_prompt) < 50:  # Sanity check
                logger.warning("⚠️ Extracted prompt is too short")
                return None

            return edited_prompt

        except Exception as e:
            logger.error(f"❌ Error extracting edited prompt: {e}")
            return None

    def compare_prompts(self, original: str, edited: str) -> Dict[str, any]:
        """Compare original and edited prompts to analyze changes."""
        return {
            "original_length": len(original),
            "edited_length": len(edited),
            "length_change": len(edited) - len(original),
            "length_change_percent": ((len(edited) - len(original)) / len(original))
            * 100,
            "significant_change": abs(len(edited) - len(original))
            > len(original) * 0.1,
        }
