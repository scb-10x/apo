"""Checkpoint management for prompt optimization."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from .config import OptimizationState, OptimizationConfig


class CheckpointManager:
    """Manages checkpointing and restoration of optimization state."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"💾 Initialized CheckpointManager with directory: {checkpoint_dir}"
        )

    def save_checkpoint(
        self,
        state: OptimizationState,
        config: OptimizationConfig,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save optimization state and config to a checkpoint.

        Args:
            state: Current optimization state
            config: Optimization configuration
            checkpoint_name: Optional custom name for checkpoint

        Returns:
            Path to the saved checkpoint file
        """
        try:
            if checkpoint_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = (
                    f"optimization_checkpoint_{timestamp}_iter_{state.iteration}"
                )

            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

            # Prepare checkpoint data
            checkpoint_data = {
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "checkpoint_version": "1.0",
                    "task": config.task,
                    "data_path": config.data_path,
                },
                "config": self._serialize_config(config),
                "state": self._serialize_state(state),
            }

            # Save to JSON file
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            logger.success(f"💾 Saved checkpoint: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            logger.error(f"❌ Error saving checkpoint: {e}")
            raise

    def load_checkpoint(
        self, checkpoint_path: str
    ) -> tuple[OptimizationState, OptimizationConfig]:
        """
        Load optimization state and config from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple of (OptimizationState, OptimizationConfig)
        """
        try:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            logger.info(f"📥 Loading checkpoint: {checkpoint_path}")

            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint version
            version = checkpoint_data.get("metadata", {}).get(
                "checkpoint_version", "unknown"
            )
            if version != "1.0":
                logger.warning(f"⚠️ Checkpoint version {version} may not be compatible")

            # Deserialize config and state
            config = self._deserialize_config(checkpoint_data["config"])
            state = self._deserialize_state(checkpoint_data["state"])

            logger.success(f"✅ Loaded checkpoint from iteration {state.iteration}")
            return state, config

        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            raise

    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """
        List available checkpoints with metadata.

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                state_data = data.get("state", {})

                checkpoint_info = {
                    "file": str(checkpoint_file),
                    "name": checkpoint_file.stem,
                    "saved_at": metadata.get("saved_at", "unknown"),
                    "task": metadata.get("task", "unknown"),
                    "iteration": state_data.get("iteration", 0),
                    "best_score": state_data.get("best_score", 0.0),
                    "total_evaluations": state_data.get("total_evaluations", 0),
                }
                checkpoints.append(checkpoint_info)

            except Exception as e:
                logger.warning(f"⚠️ Could not read checkpoint {checkpoint_file}: {e}")
                continue

        # Sort by iteration (most recent first)
        checkpoints.sort(key=lambda x: x["iteration"], reverse=True)

        logger.info(f"📋 Found {len(checkpoints)} checkpoints")
        return checkpoints

    def get_latest_checkpoint(self, task: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the latest checkpoint, optionally filtered by task.

        Args:
            task: Optional task filter

        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints()

        if task:
            checkpoints = [cp for cp in checkpoints if cp["task"] == task]

        if not checkpoints:
            logger.info("📭 No checkpoints found")
            return None

        latest = checkpoints[0]  # Already sorted by iteration
        logger.info(
            f"📌 Latest checkpoint: {latest['name']} (iteration {latest['iteration']})"
        )
        return latest["file"]

    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """
        Remove old checkpoints, keeping only the most recent ones.

        Args:
            keep_count: Number of checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_count:
            logger.info(
                f"📋 Only {len(checkpoints)} checkpoints found, no cleanup needed"
            )
            return 0

        to_delete = checkpoints[keep_count:]
        deleted_count = 0

        for checkpoint in to_delete:
            try:
                Path(checkpoint["file"]).unlink()
                deleted_count += 1
                logger.debug(f"🗑️ Deleted checkpoint: {checkpoint['name']}")
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not delete checkpoint {checkpoint['name']}: {e}"
                )

        logger.info(f"🧹 Cleaned up {deleted_count} old checkpoints")
        return deleted_count

    def _serialize_config(self, config: OptimizationConfig) -> Dict[str, Any]:
        """Serialize OptimizationConfig to dictionary."""
        return {
            "task": config.task,
            "data_path": config.data_path,
            "output_path": config.output_path,
            "prompt_types": config.prompt_types,
            "initial_function_prompt": config.initial_function_prompt,
            "initial_dialogue_prompt": config.initial_dialogue_prompt,
            "score_threshold": config.score_threshold,
            "max_iterations": config.max_iterations,
            "min_improvement_threshold": config.min_improvement_threshold,
            "sample_size": config.sample_size,
            "gradient_model": config.gradient_model,
            "gradient_temperature": config.gradient_temperature,
            "editor_model": config.editor_model,
            "editor_temperature": config.editor_temperature,
            "evaluator_model": config.evaluator_model,
            "evaluator_temperature": config.evaluator_temperature,
            "enable_checkpointing": config.enable_checkpointing,
            "checkpoint_dir": config.checkpoint_dir,
            "checkpoint_interval": config.checkpoint_interval,
            "enable_wandb": config.enable_wandb,
            "wandb_project": config.wandb_project,
            "wandb_run_name": config.wandb_run_name,
            "wandb_config": config.wandb_config,
            "batch_size": config.batch_size,
            "max_gradient_samples": config.max_gradient_samples,
            "prompt_edit_strategy": config.prompt_edit_strategy,
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> OptimizationConfig:
        """Deserialize dictionary to OptimizationConfig."""
        return OptimizationConfig(**data)

    def _serialize_state(self, state: OptimizationState) -> Dict[str, Any]:
        """Serialize OptimizationState to dictionary."""
        return {
            "iteration": state.iteration,
            "best_score": state.best_score,
            "best_function_prompt": state.best_function_prompt,
            "best_dialogue_prompt": state.best_dialogue_prompt,
            "current_function_prompt": state.current_function_prompt,
            "current_dialogue_prompt": state.current_dialogue_prompt,
            "iteration_results": [
                {
                    "iteration": result.iteration,
                    "function_prompt": result.function_prompt,
                    "dialogue_prompt": result.dialogue_prompt,
                    "score": result.score,
                    "response_score": result.response_score,
                    "functions_score": result.functions_score,
                    "total_samples": result.total_samples,
                    "failed_samples": result.failed_samples,
                    "improvement": result.improvement,
                    "timestamp": result.timestamp,
                    "gradient_count": result.gradient_count,
                }
                for result in state.iteration_results
            ],
            "total_evaluations": state.total_evaluations,
            "start_time": state.start_time,
            "last_checkpoint_iteration": state.last_checkpoint_iteration,
        }

    def _deserialize_state(self, data: Dict[str, Any]) -> OptimizationState:
        """Deserialize dictionary to OptimizationState."""
        from .config import IterationResult

        iteration_results = []
        for result_data in data.get("iteration_results", []):
            iteration_results.append(IterationResult(**result_data))

        return OptimizationState(
            iteration=data.get("iteration", 0),
            best_score=data.get("best_score", 0.0),
            best_function_prompt=data.get("best_function_prompt"),
            best_dialogue_prompt=data.get("best_dialogue_prompt"),
            current_function_prompt=data.get("current_function_prompt"),
            current_dialogue_prompt=data.get("current_dialogue_prompt"),
            iteration_results=iteration_results,
            total_evaluations=data.get("total_evaluations", 0),
            start_time=data.get("start_time"),
            last_checkpoint_iteration=data.get("last_checkpoint_iteration", -1),
        )
