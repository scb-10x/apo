"""
Prompt optimization CLI using cyclopts.

Supports both command-line arguments and configuration files.
"""

import sys
from pathlib import Path
from typing import Optional, List, Literal, Annotated
from loguru import logger
import cyclopts

from optimization.config import OptimizationConfig
from optimization.optimizer import PromptOptimizer
from optimization.checkpoint_manager import CheckpointManager

app = cyclopts.App(
    name="optimize_prompts", help="Advanced prompt optimization for AI agents"
)


@app.command
def run(
    # Optional configuration file
    config: Annotated[
        Optional[str], cyclopts.Parameter(help="Path to configuration file (YAML/JSON)")
    ] = None,
    # Parameters (required if no config provided)
    task: Annotated[
        Optional[Literal["task1", "task2", "both"]],
        cyclopts.Parameter(help="Task to optimize for"),
    ] = None,
    data: Annotated[
        Optional[str], cyclopts.Parameter(help="Path to the dataset file")
    ] = None,
    output: Annotated[
        Optional[str], cyclopts.Parameter(help="Output path for results")
    ] = None,
    # Prompt configuration
    prompt_types: Annotated[
        List[Literal["function", "dialogue"]],
        cyclopts.Parameter(help="Types of prompts to optimize"),
    ] = ["function"],
    initial_function_prompt: Annotated[
        Optional[str], cyclopts.Parameter(help="Initial function prompt (optional)")
    ] = None,
    initial_dialogue_prompt: Annotated[
        Optional[str], cyclopts.Parameter(help="Initial dialogue prompt (optional)")
    ] = None,
    # Optimization parameters
    score_threshold: Annotated[
        float, cyclopts.Parameter(help="Target score threshold")
    ] = 0.8,
    max_iterations: Annotated[
        int, cyclopts.Parameter(help="Maximum optimization iterations")
    ] = 10,
    min_improvement: Annotated[
        float, cyclopts.Parameter(help="Minimum improvement threshold for convergence")
    ] = 0.01,
    sample_size: Annotated[
        Optional[int],
        cyclopts.Parameter(help="Sample size for evaluation (None = use all)"),
    ] = None,
    # Model configuration
    gradient_model: Annotated[
        str, cyclopts.Parameter(help="Model for gradient generation")
    ] = "gpt-4o-mini",
    gradient_temperature: Annotated[
        float, cyclopts.Parameter(help="Temperature for gradient generation")
    ] = 0.7,
    editor_model: Annotated[
        str, cyclopts.Parameter(help="Model for prompt editing")
    ] = "gpt-4o-mini",
    editor_temperature: Annotated[
        float, cyclopts.Parameter(help="Temperature for prompt editing")
    ] = 0.3,
    evaluator_model: Annotated[
        str, cyclopts.Parameter(help="Model for evaluation")
    ] = "gpt-4o-mini",
    evaluator_temperature: Annotated[
        float, cyclopts.Parameter(help="Temperature for evaluation")
    ] = 0.1,
    # Advanced parameters
    batch_size: Annotated[
        int, cyclopts.Parameter(help="Batch size for gradient analysis")
    ] = 5,
    max_gradient_samples: Annotated[
        int, cyclopts.Parameter(help="Maximum failed samples for gradient generation")
    ] = 20,
    edit_strategy: Annotated[
        Literal["incremental", "replacement"],
        cyclopts.Parameter(help="Prompt editing strategy"),
    ] = "incremental",
    # Advanced optimization features
    enable_gradient_mini_batch: Annotated[
        bool, cyclopts.Parameter(help="Enable mini-batch gradient generation")
    ] = False,
    gradient_mini_batch_size: Annotated[
        int, cyclopts.Parameter(help="Size of mini-batch for gradient generation")
    ] = 10,
    enable_prompt_candidates: Annotated[
        bool,
        cyclopts.Parameter(
            help="Enable multiple prompt candidates with Monte Carlo selection"
        ),
    ] = False,
    num_prompt_candidates: Annotated[
        int, cyclopts.Parameter(help="Number of prompt candidates to generate")
    ] = 3,
    monte_carlo_sample_size: Annotated[
        int,
        cyclopts.Parameter(help="Sample size for Monte Carlo evaluation of candidates"),
    ] = 10,
    enable_beam_search: Annotated[
        bool, cyclopts.Parameter(help="Enable beam search mode")
    ] = False,
    beam_width: Annotated[int, cyclopts.Parameter(help="Width of beam search")] = 3,
    enable_gradient_memory: Annotated[
        bool, cyclopts.Parameter(help="Enable gradient memory across iterations")
    ] = False,
    gradient_memory_size: Annotated[
        int,
        cyclopts.Parameter(
            help="Number of past iterations to remember in gradient memory"
        ),
    ] = 5,
    # Performance optimization
    n_parallel: Annotated[
        int, cyclopts.Parameter(help="Number of parallel LM evaluation requests")
    ] = 16,
    # Checkpoint configuration
    no_checkpointing: Annotated[
        bool, cyclopts.Parameter(help="Disable checkpointing")
    ] = False,
    checkpoint_dir: Annotated[
        str, cyclopts.Parameter(help="Directory for checkpoints")
    ] = "checkpoints",
    checkpoint_interval: Annotated[
        int, cyclopts.Parameter(help="Save checkpoint every N iterations")
    ] = 1,
    # Wandb configuration
    wandb: Annotated[bool, cyclopts.Parameter(help="Enable Wandb logging")] = False,
    wandb_project: Annotated[
        Optional[str], cyclopts.Parameter(help="Wandb project name")
    ] = None,
    wandb_run_name: Annotated[
        Optional[str], cyclopts.Parameter(help="Wandb run name")
    ] = None,
    # Other options
    verbose: Annotated[bool, cyclopts.Parameter(help="Enable verbose logging")] = False,
) -> None:
    """Run prompt optimization."""

    # Setup logging
    setup_logging(verbose)

    try:
        # Validate required parameters
        if config:
            # Load configuration from file
            logger.info(f"📁 Loading configuration from: {config}")
            optimization_config = OptimizationConfig.from_file(config)
            logger.info("✅ Configuration loaded successfully")

            # Override config values with CLI arguments if provided
            if task is not None:
                optimization_config.task = task
            if data is not None:
                optimization_config.data_path = data
            if output is not None:
                optimization_config.output_path = output

            # Override other CLI parameters if provided (non-None values)
            if prompt_types != ["function"]:  # Default value check
                optimization_config.prompt_types = prompt_types
            if initial_function_prompt is not None:
                optimization_config.initial_function_prompt = initial_function_prompt
            if initial_dialogue_prompt is not None:
                optimization_config.initial_dialogue_prompt = initial_dialogue_prompt
            if score_threshold != 0.8:  # Default value check
                optimization_config.score_threshold = score_threshold
            if max_iterations != 10:  # Default value check
                optimization_config.max_iterations = max_iterations
            if min_improvement != 0.01:  # Default value check
                optimization_config.min_improvement_threshold = min_improvement
            if sample_size is not None:
                optimization_config.sample_size = sample_size
            if gradient_model != "gpt-4o-mini":  # Default value check
                optimization_config.gradient_model = gradient_model
            if gradient_temperature != 0.7:  # Default value check
                optimization_config.gradient_temperature = gradient_temperature
            if editor_model != "gpt-4o-mini":  # Default value check
                optimization_config.editor_model = editor_model
            if editor_temperature != 0.3:  # Default value check
                optimization_config.editor_temperature = editor_temperature
            if evaluator_model != "gpt-4o-mini":  # Default value check
                optimization_config.evaluator_model = evaluator_model
            if evaluator_temperature != 0.1:  # Default value check
                optimization_config.evaluator_temperature = evaluator_temperature
            if batch_size != 5:  # Default value check
                optimization_config.batch_size = batch_size
            if max_gradient_samples != 20:  # Default value check
                optimization_config.max_gradient_samples = max_gradient_samples
            if edit_strategy != "incremental":  # Default value check
                optimization_config.prompt_edit_strategy = edit_strategy
            if enable_gradient_mini_batch:  # Default is False
                optimization_config.enable_gradient_mini_batch = (
                    enable_gradient_mini_batch
                )
            if gradient_mini_batch_size != 10:  # Default value check
                optimization_config.gradient_mini_batch_size = gradient_mini_batch_size
            if enable_prompt_candidates:  # Default is False
                optimization_config.enable_prompt_candidates = enable_prompt_candidates
            if num_prompt_candidates != 3:  # Default value check
                optimization_config.num_prompt_candidates = num_prompt_candidates
            if monte_carlo_sample_size != 10:  # Default value check
                optimization_config.monte_carlo_sample_size = monte_carlo_sample_size
            if enable_beam_search:  # Default is False
                optimization_config.enable_beam_search = enable_beam_search
            if beam_width != 3:  # Default value check
                optimization_config.beam_width = beam_width
            if enable_gradient_memory:  # Default is False
                optimization_config.enable_gradient_memory = enable_gradient_memory
            if gradient_memory_size != 5:  # Default value check
                optimization_config.gradient_memory_size = gradient_memory_size
            if n_parallel != 16:  # Default value check
                optimization_config.n_parallel = n_parallel
            if no_checkpointing:  # Default is False
                optimization_config.enable_checkpointing = not no_checkpointing
            if checkpoint_dir != "checkpoints":  # Default value check
                optimization_config.checkpoint_dir = checkpoint_dir
            if checkpoint_interval != 1:  # Default value check
                optimization_config.checkpoint_interval = checkpoint_interval
            if wandb:  # Default is False
                optimization_config.enable_wandb = wandb
            if wandb_project is not None:
                optimization_config.wandb_project = wandb_project
            if wandb_run_name is not None:
                optimization_config.wandb_run_name = wandb_run_name

        else:
            # No config file provided, validate required CLI arguments
            if task is None:
                raise ValueError("--task is required when no config file is provided")
            if data is None:
                raise ValueError("--data is required when no config file is provided")
            if output is None:
                raise ValueError("--output is required when no config file is provided")

            # Create configuration from CLI arguments
            optimization_config = OptimizationConfig(
                task=task,
                data_path=data,
                output_path=output,
                prompt_types=prompt_types,
                initial_function_prompt=initial_function_prompt,
                initial_dialogue_prompt=initial_dialogue_prompt,
                score_threshold=score_threshold,
                max_iterations=max_iterations,
                min_improvement_threshold=min_improvement,
                sample_size=sample_size,
                gradient_model=gradient_model,
                gradient_temperature=gradient_temperature,
                editor_model=editor_model,
                editor_temperature=editor_temperature,
                evaluator_model=evaluator_model,
                evaluator_temperature=evaluator_temperature,
                enable_checkpointing=not no_checkpointing,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=checkpoint_interval,
                enable_wandb=wandb,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name,
                batch_size=batch_size,
                max_gradient_samples=max_gradient_samples,
                prompt_edit_strategy=edit_strategy,
                # Advanced optimization features
                enable_gradient_mini_batch=enable_gradient_mini_batch,
                gradient_mini_batch_size=gradient_mini_batch_size,
                enable_prompt_candidates=enable_prompt_candidates,
                num_prompt_candidates=num_prompt_candidates,
                monte_carlo_sample_size=monte_carlo_sample_size,
                enable_beam_search=enable_beam_search,
                beam_width=beam_width,
                enable_gradient_memory=enable_gradient_memory,
                gradient_memory_size=gradient_memory_size,
                n_parallel=n_parallel,
            )

        # Log configuration summary
        logger.info("🎯 Optimization Configuration:")
        logger.info(f"   Task: {optimization_config.task}")
        logger.info(f"   Prompt types: {optimization_config.prompt_types}")
        logger.info(f"   Score threshold: {optimization_config.score_threshold}")
        logger.info(f"   Max iterations: {optimization_config.max_iterations}")

        # Log advanced features if enabled
        advanced_features = []
        if optimization_config.enable_gradient_mini_batch:
            advanced_features.append(
                f"Mini-batch ({optimization_config.gradient_mini_batch_size})"
            )
        if optimization_config.enable_prompt_candidates:
            advanced_features.append(
                f"Candidates ({optimization_config.num_prompt_candidates})"
            )
        if optimization_config.enable_beam_search:
            advanced_features.append(f"Beam search ({optimization_config.beam_width})")
        if optimization_config.enable_gradient_memory:
            advanced_features.append(
                f"Memory ({optimization_config.gradient_memory_size})"
            )

        if advanced_features:
            logger.info(f"   Advanced features: {', '.join(advanced_features)}")

        # Create and run optimizer
        optimizer = PromptOptimizer(optimization_config)
        final_state = optimizer.optimize()

        # Print final results
        print("\n" + "=" * 60)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Final Score: {final_state.best_score:.3f}")
        print(f"🔄 Total Iterations: {final_state.iteration}")
        print(f"📋 Total Evaluations: {final_state.total_evaluations}")

        if final_state.iteration_results:
            initial_score = final_state.iteration_results[0].score
            improvement = final_state.best_score - initial_score
            print(f"📈 Total Improvement: +{improvement:.3f}")

        threshold_reached = (
            final_state.best_score >= optimization_config.score_threshold
        )
        print(f"🎯 Threshold Reached: {'✅ Yes' if threshold_reached else '❌ No'}")

        if final_state.best_function_prompt:
            print(f"💾 Results saved to: {optimization_config.output_path}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        sys.exit(1)


@app.command
def resume(
    checkpoint: Annotated[
        str, cyclopts.Parameter(help="Path to checkpoint file to resume from")
    ],
    verbose: Annotated[bool, cyclopts.Parameter(help="Enable verbose logging")] = False,
) -> None:
    """Resume optimization from a checkpoint."""

    # Setup logging
    setup_logging(verbose)

    try:
        logger.info(f"📥 Resuming optimization from: {checkpoint}")

        # Load config from checkpoint
        checkpoint_manager = CheckpointManager()
        state, config = checkpoint_manager.load_checkpoint(checkpoint)

        logger.info(f"✅ Resumed from iteration {state.iteration}")
        logger.info(f"📊 Current best score: {state.best_score:.3f}")

        # Create optimizer with loaded config
        optimizer = PromptOptimizer(config)
        optimizer.state = state

        # Continue optimization
        final_state = optimizer.optimize()

        # Print final results
        print("\n" + "=" * 60)
        print("🎉 RESUMED OPTIMIZATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Final Score: {final_state.best_score:.3f}")
        print(f"🔄 Total Iterations: {final_state.iteration}")
        print(f"📋 Total Evaluations: {final_state.total_evaluations}")

    except Exception as e:
        logger.error(f"❌ Resume failed: {e}")
        sys.exit(1)


@app.command
def list_checkpoints(
    checkpoint_dir: Annotated[
        str, cyclopts.Parameter(help="Directory to list checkpoints from")
    ] = "checkpoints",
) -> None:
    """List available checkpoints."""

    try:
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        checkpoints = checkpoint_manager.list_checkpoints()

        if not checkpoints:
            print("📭 No checkpoints found")
            return

        print(f"📋 Found {len(checkpoints)} checkpoints:")
        print()

        for cp in checkpoints:
            print(f"📌 {cp['name']}")
            print(f"   📅 Saved: {cp['saved_at']}")
            print(f"   🎯 Task: {cp['task']}")
            print(f"   🔄 Iteration: {cp['iteration']}")
            print(f"   📊 Best Score: {cp['best_score']:.3f}")
            print(f"   📋 Evaluations: {cp['total_evaluations']}")
            print(f"   📁 File: {cp['file']}")
            print()

    except Exception as e:
        logger.error(f"❌ Failed to list checkpoints: {e}")
        sys.exit(1)


@app.command
def generate_config(
    output: Annotated[
        str, cyclopts.Parameter(help="Output path for the configuration file")
    ] = "config.yaml",
    format: Annotated[
        Literal["yaml", "json"], cyclopts.Parameter(help="Configuration file format")
    ] = "yaml",
    include_advanced: Annotated[
        bool, cyclopts.Parameter(help="Include advanced optimization features")
    ] = True,
) -> None:
    """Generate an example configuration file."""

    try:
        # Create example configuration
        if include_advanced:
            config = OptimizationConfig(
                # Required parameters
                task="both",
                data_path="data/dataset.json",
                output_path="results/optimization",
                # Prompt configuration
                prompt_types=["function", "dialogue"],
                initial_function_prompt=None,
                initial_dialogue_prompt=None,
                # Optimization parameters
                score_threshold=0.85,
                max_iterations=15,
                min_improvement_threshold=0.01,
                sample_size=None,
                # Model configuration
                gradient_model="gpt-4o-mini",
                gradient_temperature=0.7,
                editor_model="gpt-4o-mini",
                editor_temperature=0.3,
                evaluator_model="gpt-4o-mini",
                evaluator_temperature=0.1,
                # Advanced parameters
                batch_size=8,
                max_gradient_samples=25,
                prompt_edit_strategy="incremental",
                # Advanced optimization features
                enable_gradient_mini_batch=True,
                gradient_mini_batch_size=12,
                enable_prompt_candidates=True,
                num_prompt_candidates=3,
                monte_carlo_sample_size=15,
                enable_beam_search=False,
                beam_width=3,
                enable_gradient_memory=True,
                gradient_memory_size=5,
                # Checkpoint configuration
                enable_checkpointing=True,
                checkpoint_dir="checkpoints",
                checkpoint_interval=1,
                # Wandb configuration
                enable_wandb=False,
                wandb_project="prompt-optimization",
                wandb_run_name=None,
            )
        else:
            # Basic configuration
            config = OptimizationConfig(
                task="task1",
                data_path="data/dataset.json",
                output_path="results/optimization",
                prompt_types=["function"],
                score_threshold=0.8,
                max_iterations=10,
            )

        # Ensure output has correct extension
        output_path = Path(output)
        if format == "yaml" and output_path.suffix.lower() not in [".yaml", ".yml"]:
            output_path = output_path.with_suffix(".yaml")
        elif format == "json" and output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")

        # Save configuration
        config.to_file(str(output_path))

        print(f"✅ Generated example configuration: {output_path}")
        print(f"📝 Format: {format.upper()}")
        print(
            f"🔧 Advanced features: {'✅ Included' if include_advanced else '❌ Basic only'}"
        )
        print()
        print("To use this configuration:")
        print(f"  optimize_prompts run --config {output_path} <task> <data> <output>")

    except Exception as e:
        logger.error(f"❌ Failed to generate config: {e}")
        sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if verbose else "INFO"

    logger.add(
        sink=lambda message: print(message, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
