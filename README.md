# APO: Automatic Prompt Optimization for Persona-based LLM Agents

**This repository is archived and released as-is. If you’re interested in this work, please contact us.**

---


[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**APO** is a research tool for automatically optimizing prompts used in creating persona-based Large Language Model (LLM) agents. This repository contains the implementation for our paper submission to the **[Commonsense Persona-grounded Dialogue Challenge 2025](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025)** at EMNLP 2025.

## 🎯 Research Objective

This project addresses the challenge of automatically optimizing prompts for persona-based LLM agents through an iterative optimization framework that leverages gradient-based feedback from language models. The goal is to improve the performance of agents in persona-grounded dialogue tasks without manual prompt engineering.

## 📋 Task Descriptions

### Task 1: Task-Oriented Dialogue Agents
Agents must correctly identify and call appropriate functions based on user requests in persona-grounded scenarios. The optimization focuses on improving function selection accuracy and parameter extraction. This task requires executing necessary functions depending on the situation while maintaining natural conversation flow.

### Task 2: Context-Aware Dialogue Agents
Agents must generate contextually appropriate and persona-consistent responses in multi-turn conversations without the need for function execution. The optimization targets response quality, persona adherence, and contextual relevance for natural, human-like interactions.

## 🚀 Key Features

### 1. **Gradient-Based Prompt Optimization**
- Automatically generates improvement feedback from failed samples
- Uses language models to analyze performance gaps
- Provides actionable suggestions for prompt refinement

### 2. **Multi-Task Support**
- **Task 1**: Function calling optimization for persona-based agents
- **Task 2**: Dialogue generation optimization
- Support for optimizing both tasks simultaneously

### 3. **Advanced Optimization Strategies**
- **Beam Search**: Explores multiple prompt candidates
- **Monte Carlo Selection**: Stochastic sampling for robust optimization
- **Gradient Memory**: Maintains optimization history across iterations
- **Mini-batch Processing**: Efficient handling of large datasets

### 4. **Robust Evaluation Framework**
- Automated evaluation using language model judges
- Comprehensive scoring across multiple dimensions
- Checkpointing for long-running optimizations

## 🏗️ Project Structure

```
apo/
├── configs/                          # Configuration files for experiments
│   └── function_calling.yaml         # Main experiment configuration
├── data/                             # Competition dataset (from starter pack)
│   ├── task1_sample.json            # Task 1 sample data
│   ├── task1_train.json             # Task 1 training data
│   ├── task2_sample.json            # Task 2 sample data
│   └── task2_train.json             # Task 2 training data
├── src/
│   ├── optimization/                 # Core optimization framework
│   │   ├── optimizer.py             # Main optimization orchestrator
│   │   ├── gradient_generator.py    # Gradient generation from failed samples
│   │   ├── prompt_editor.py         # Prompt editing based on gradients
│   │   ├── config.py                # Configuration management
│   │   ├── checkpoint_manager.py    # Optimization state persistence
│   │   └── report_generator.py      # Results and analysis generation
│   ├── optimize_prompts.py          # CLI entry point for optimization
│   ├── agents/                      # LLM agent implementations
│   ├── function_calls/              # Function calling utilities
│   ├── tasks/                       # Task runners (from competition starter pack)
│   └── npcdataset/                  # Dataset utilities (from starter pack)
├── env.example                      # Environment configuration template
├── pyproject.toml                   # Project dependencies and metadata
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.12 or higher
- OpenAI API key (or other supported LLM provider)

### Setup
```bash
# Clone the repository
git clone https://github.com/scb-10x/apo.git
cd apo

# Install dependencies using uv
uv sync
```

### Environment Configuration
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual API keys and configuration
# Required: OPENAI_API_KEY
# Optional: Adjust model settings, optimization parameters, and other configurations
```

**Important**: Never commit your `.env` file to version control. The `env.example` file contains all available configuration options with sensible defaults.

## 🚀 Quick Start

```bash
# Run optimization with default configuration
uv run src/optimize_prompts.py run --task task1 --data data/task1_train.json --output results/

# Run with custom configuration file
uv run src/optimize_prompts.py run --config configs/function_calling.yaml
```

## 🔬 Research Methodology

### Optimization Loop
1. **Evaluation**: Assess current prompt performance on target task
2. **Gradient Generation**: Analyze failed samples to identify improvement areas
3. **Prompt Editing**: Apply gradient-based feedback to refine prompts
4. **Validation**: Test improved prompts and measure performance gains
5. **Iteration**: Repeat until convergence or maximum iterations reached

### Key Innovations
- **Language Model Gradients**: Uses LLMs to generate improvement feedback
- **Multi-Strategy Optimization**: Combines beam search, Monte Carlo, and memory
- **Automated Evaluation**: Eliminates need for manual prompt assessment
- **Checkpointing**: Enables resumption of long-running optimizations

## 📖 Usage Guide

### Configuration Options
The optimization can be configured through YAML files or command-line arguments:

```yaml
# configs/function_calling.yaml
task: "task1"
data_path: "data/task1_train.json"
output_path: "results/function_calling"

# Optimization parameters
score_threshold: 0.95
max_iterations: 30
min_improvement_threshold: 0.01

# Model configuration
gradient_model: "gpt-4.1-mini"
editor_model: "gpt-4.1"
evaluator_model: "gpt-4.1-mini"

# Advanced features
enable_beam_search: true
beam_width: 3
enable_gradient_memory: true
```

### Command-Line Interface
```bash
# View all available options
uv run src/optimize_prompts.py run --help

# Run with specific parameters
uv run src/optimize_prompts.py run \
    --task task1 \
    --data data/task1_train.json \
    --output results/ \
    --score-threshold 0.9 \
    --max-iterations 20 \
    --gradient-model gpt-4o-mini
```

### Environment Variables
The project uses environment variables for configuration. Copy `env.example` to `.env` and customize the settings:

- **Required**: `OPENAI_API_KEY` for LLM access
- **Optional**: Model selection, optimization parameters, logging settings
- **Advanced**: Performance tuning, feature toggles, and debugging options

See `env.example` for the complete list of available configuration options.

## 🔧 Advanced Configuration

### Custom Prompt Types
```yaml
prompt_types:
  - "function"      # Function calling prompts
  - "dialogue"      # Dialogue generation prompts
```

### Optimization Strategies
```yaml
# Enable advanced optimization features
enable_beam_search: true
enable_gradient_memory: true
enable_prompt_candidates: true
enable_gradient_mini_batch: true
```

### Model Configuration
```yaml
# Different models for different optimization stages
gradient_model: "gpt-4o-mini"      # For gradient generation
editor_model: "gpt-4o"             # For prompt editing
evaluator_model: "gpt-4o-mini"     # For evaluation
```

## 📊 Results and Analysis

The optimization process generates comprehensive reports including:
- Performance metrics across iterations
- Gradient analysis and improvement patterns
- Prompt evolution tracking
- Failure case analysis
- Optimization convergence statistics

Results are saved in the specified output directory with detailed logging and checkpoint files.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **[Commonsense Persona-grounded Dialogue Challenge 2025](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025)**: For organizing this shared task at EMNLP 2025 and providing the starter pack and dataset
- **[Microsoft LMOps](https://github.com/microsoft/LMOps/tree/main/prompt_optimization)**: For the original Prompt Optimization with Textual Gradients (ProTeGi) implementation that inspired some of our optimization approaches

---

**Note**: The `data/` directory and `src/tasks/`, `src/agents/`, `src/function_calls/`, and `src/npcdataset/` modules are from the competition's starter pack. The core research contribution is in the `src/optimization/` module and `src/optimize_prompts.py`.
