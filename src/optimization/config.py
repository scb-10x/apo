"""Configuration classes for prompt optimization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path
import yaml
import json


@dataclass
class OptimizationConfig:
    """Configuration for a prompt optimization run."""

    # Required configurations
    task: Literal["task1", "task2", "both"]
    data_path: str
    output_path: str

    # Prompt configuration
    prompt_types: List[Literal["function", "dialogue"]] = field(
        default_factory=lambda: ["function"]
    )
    initial_function_prompt: Optional[str] = None
    initial_dialogue_prompt: Optional[str] = None

    # Optimization parameters
    score_threshold: float = 0.8
    max_iterations: int = 10
    min_improvement_threshold: float = 0.01
    sample_size: Optional[int] = None  # None = use all data

    # LM configurations
    gradient_model: str = "gpt-4o-mini"
    gradient_temperature: float = 0.7
    editor_model: str = "gpt-4o-mini"
    editor_temperature: float = 0.3
    evaluator_model: str = "gpt-4o-mini"
    evaluator_temperature: float = 0.1

    # Checkpoint configuration
    enable_checkpointing: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1  # Save every N iterations

    # Logging configuration
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_config: Dict[str, Any] = field(default_factory=dict)

    # Advanced parameters
    batch_size: int = 5  # Number of samples to analyze for gradients
    max_gradient_samples: int = 20  # Max failed samples to use for gradient generation
    prompt_edit_strategy: Literal["incremental", "replacement"] = "incremental"

    # Advanced optimization features
    # Mini-batch gradient generation
    enable_gradient_mini_batch: bool = False
    gradient_mini_batch_size: int = 10

    # Multiple prompt candidates with Monte Carlo selection
    enable_prompt_candidates: bool = False
    num_prompt_candidates: int = 3
    monte_carlo_sample_size: int = 10

    # Beam search mode
    enable_beam_search: bool = False
    beam_width: int = 3

    # Gradient memory across iterations
    enable_gradient_memory: bool = False
    gradient_memory_size: int = 5  # Number of past iterations to remember

    # Performance optimization
    n_parallel: int = 16  # Number of parallel LM evaluation requests

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.task not in ["task1", "task2", "both"]:
            raise ValueError(f"Invalid task: {self.task}")

        if not self.prompt_types:
            raise ValueError("At least one prompt type must be specified")

        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        # Create output directory if it doesn't exist
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create checkpoint directory if checkpointing is enabled
        if self.enable_checkpointing:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if self.score_threshold < 0 or self.score_threshold > 1:
            raise ValueError("Score threshold must be between 0 and 1")

        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")

    @classmethod
    def from_file(cls, config_path: str) -> "OptimizationConfig":
        """
        Load configuration from a YAML or JSON file.

        Args:
            config_path: Path to the configuration file

        Returns:
            OptimizationConfig instance
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise ValueError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_file.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_file.suffix}"
                )

        return cls(**config_data)

    def to_file(self, config_path: str) -> None:
        """
        Save configuration to a YAML or JSON file.

        Args:
            config_path: Path where to save the configuration file
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict
        config_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

        with open(config_file, "w", encoding="utf-8") as f:
            if config_file.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_file.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_file.suffix}"
                )


@dataclass
class IterationResult:
    """Results from a single optimization iteration."""

    iteration: int
    function_prompt: Optional[str] = None
    dialogue_prompt: Optional[str] = None
    score: float = 0.0
    response_score: float = 0.0
    functions_score: float = 0.0
    total_samples: int = 0
    failed_samples: int = 0
    improvement: float = 0.0
    timestamp: str = ""
    gradient_count: int = 0


@dataclass
class PromptCandidate:
    """A prompt candidate in beam search or candidate generation."""

    function_prompt: Optional[str] = None
    dialogue_prompt: Optional[str] = None
    score: float = 0.0
    evaluation_count: int = 0
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradientMemory:
    """Memory of gradients from previous iterations."""

    iteration: int
    gradients: List[Any]  # List of Gradient objects
    gradient_summary: str
    timestamp: str


@dataclass
class OptimizationState:
    """Current state of the optimization process."""

    iteration: int = 0
    best_score: float = 0.0
    best_function_prompt: Optional[str] = None
    best_dialogue_prompt: Optional[str] = None
    current_function_prompt: Optional[str] = None
    current_dialogue_prompt: Optional[str] = None
    iteration_results: List[IterationResult] = field(default_factory=list)
    total_evaluations: int = 0
    start_time: Optional[str] = None
    last_checkpoint_iteration: int = -1

    # Advanced features state
    beam_candidates: List[PromptCandidate] = field(default_factory=list)
    gradient_memory: List[GradientMemory] = field(default_factory=list)
