# Cerebras CLI

A command-line interface for calling the Cerebras OpenAI-compatible chat API with built-in comparison and scoring capabilities.

## Features

- **Dual API Support**: Compare Cerebras (Qwen 3 235B Instruct) and Anthropic (Claude 3.5 Haiku) generation side-by-side
- **Automatic Scoring**: Code quality scoring using a comprehensive rubric
- **Performance Metrics**: Track TTFT, latency, and throughput
- **Batch Processing**: Process multiple prompts from JSON/JSONL files
- **Detailed Reports**: All results saved to files with comprehensive details

## Installation

This project uses [UV](https://github.com/astral-sh/uv) for package management. Install UV first:

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then initialize the project and install dependencies:

```bash
# Initialize the project with UV
uv init

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (UV will automatically install from requirements.txt)
uv sync
```

## Setup

Set your API keys as environment variables:

```bash
export CEREBRAS_API_KEY="your-cerebras-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"  # Required for comparison and scoring
```

## Default Behavior

By default, the script:
- **Compares** Cerebras (Qwen 3 235B Instruct) vs Anthropic (Claude 3.5 Haiku) generation (`--compare` enabled)
- **Scores** all generated code (`--score` enabled)
- **Saves** all outputs to files (not printed to console)

To disable these defaults:
```bash
uv run python cerebras_chat.py "prompt" --no-compare --no-score
```

## Usage

All examples below assume you have activated the virtual environment (`source .venv/bin/activate`) or are using `uv run`:

### Basic Usage (Single Prompt)

```bash
uv run python cerebras_chat.py "Implement LRU cache in Python"
```

This will automatically:
- Generate code with both Cerebras (Qwen 3 235B Instruct) and Anthropic (Claude 3.5 Haiku)
- Score both outputs
- Display comparison tables
- Save detailed results to files

### With Custom Model and Temperature

```bash
uv run python cerebras_chat.py "Write a function" --model qwen-3-235b-a22b-instruct-2507 --temperature 0.7
```

### With System Message

```bash
uv run python cerebras_chat.py "Hello!" --system "You are a helpful coding assistant"
```

### Read from Stdin

```bash
echo "What is Python?" | uv run python cerebras_chat.py
```

### Get Full JSON Response

```bash
uv run python cerebras_chat.py "Hello!" --json
```

### Batch Processing from JSON/JSONL File

```bash
# Using UV (recommended)
uv run python cerebras_chat.py --input-file eval.jsonl

# Or with activated virtual environment
source .venv/bin/activate
python cerebras_chat.py --input-file eval.jsonl
```

This will:
1. Read all prompts from the JSONL file
2. Process each prompt through the comparison pipeline
3. Save individual results to separate files
4. Generate a comprehensive summary

**File Format:**
Each line should be a JSON object:
```json
{"prompt": "Implement LRU cache in Python", "language": "python"}
{"prompt": "Create a thread-safe queue", "language": "rust"}
```

**Output Structure:**
```
batch_output_YYYYMMDD_HHMMSS/
├── prompt_001_cerebras.txt
├── prompt_001_anthropic.txt
├── prompt_002_cerebras.txt
├── prompt_002_anthropic.txt
├── ...
└── summary_YYYYMMDD_HHMMSS.txt
```

Each individual file contains:
- Prompt information
- Performance metrics (TTFT, latency, throughput)
- Code quality scores (Correctness, Code Quality, Efficiency, Documentation)
- Feedback from scorer
- Generated code

The summary file includes:
- Overall statistics (average scores, performance metrics)
- Win/loss breakdown
- Detailed results for each prompt

## Command-Line Options

### Input Options
- `prompt`: User prompt (positional argument, or read from stdin)
- `--input-file`: JSON/JSONL file with list of prompts for batch processing

### Model Options
- `--model`: Cerebras model name (default: `qwen-3-235b-a22b-instruct-2507`)
- `--anthropic-model`: Anthropic model to use (default: `claude-3-5-haiku-20241022`)
- `--system`: System message to set assistant behavior
- `--temperature`: Sampling temperature (0.0 = deterministic, default: 0.0)
- `--max-tokens`: Maximum completion tokens (-1 = no limit, default: -1)
- `--top-p`: Nucleus sampling parameter (default: 1.0)
- `--seed`: Random seed for reproducibility (default: 0)
- `--stream`: Stream responses (not yet implemented)

### API Options
- `--api-key`: Cerebras API key (default: from `CEREBRAS_API_KEY` env var)

### Output Options
- `--json`: Output full JSON response instead of just content

### Scoring Options (Enabled by Default)
- `--score`: Score the generated code using a rubric (default: enabled)
- `--no-score`: Disable scoring
- `--language`: Programming language for scoring (default: python)
- `--score-model`: Model to use for scoring (default: Anthropic claude-3-5-haiku-20241022)

**Scoring Rubric:**
- **Correctness** (0.30): Does the code correctly implement requirements?
- **Code Quality** (0.30): Is the code clean, readable, well-structured?
- **Efficiency** (0.20): Is the code efficient and optimal?
- **Documentation** (0.20): Are there docstrings, comments, type hints?

### Comparison Options (Enabled by Default)
- `--compare`: Compare Cerebras (Qwen 3 235B Instruct) vs Anthropic (Claude 3.5 Haiku) generation (default: enabled)
- `--no-compare`: Disable comparison mode (use single API only)

**Comparison includes:**
- Side-by-side performance metrics with speedup calculations
- Code quality score comparison
- Winner identification for each metric
- Comprehensive summary

## Examples

### Single Prompt Examples

```bash
# Simple query (automatically compares and scores)
python cerebras_chat.py "What is machine learning?"

# Code generation with system prompt
python cerebras_chat.py "Implement a binary search" \
    --system "You are an expert Python programmer" \
    --temperature 0.3

# Generate and score code (explicit, though enabled by default)
python cerebras_chat.py "Implement LRU cache in Python" --score --language python

# Compare with custom models
python cerebras_chat.py "Write a function" \
    --model qwen-3-235b-a22b-instruct-2507 \
    --anthropic-model claude-3-5-haiku-20241022

# Disable comparison (single API only)
python cerebras_chat.py "Implement LRU cache" --no-compare

# Disable scoring
python cerebras_chat.py "Implement LRU cache" --no-score

# Multiple lines from stdin
cat prompt.txt | python cerebras_chat.py

# Get structured JSON output
python cerebras_chat.py "List prime numbers" --json
```

### Batch Processing Examples

```bash
# Process all prompts from eval.jsonl
python cerebras_chat.py --input-file eval.jsonl

# Batch process with custom language
python cerebras_chat.py --input-file eval.jsonl --language rust
```

## Output Files

### Single Prompt Mode

When processing a single prompt, files are saved as:
- `cerebras_output_YYYYMMDD_HHMMSS.txt`
- `anthropic_output_YYYYMMDD_HHMMSS.txt`

Each file contains:
- Prompt and model information
- Performance metrics (TTFT, latency, throughput, tokens)
- Code quality scores (with weighted calculations)
- Feedback from scorer
- Generated code

### Batch Processing Mode

When using `--input-file`, all outputs are saved to:
- `batch_output_YYYYMMDD_HHMMSS/`

Individual files:
- `prompt_001_cerebras.txt`, `prompt_001_anthropic.txt`
- `prompt_002_cerebras.txt`, `prompt_002_anthropic.txt`
- ...

Summary file:
- `summary_YYYYMMDD_HHMMSS.txt` - Comprehensive summary with statistics

## Comparison Output

The script displays two comparison tables:

### 1. Performance Comparison Table
Shows side-by-side metrics with speedup calculations:
- Time To First Token (TTFT)
- Latency Between Tokens
- Input/Output Throughput
- Total Time
- Token counts
- **Speed Up** column showing how many times faster the winner is

### 2. Code Quality Score Comparison Table
Shows side-by-side scores with weighted calculations:
- Correctness (weight: 0.30)
- Code Quality (weight: 0.30)
- Efficiency (weight: 0.20)
- Documentation (weight: 0.20)
- TOTAL SCORE
- Winner for each metric

### 3. Comprehensive Summary
Includes:
- Performance summary with key metrics
- Quality summary with detailed breakdown
- Quality winner identification
- Feedback from both providers

## Example Output

### Batch Processing Summary

When processing multiple prompts, the script displays a comprehensive summary with both code quality scores and performance metrics:

```
=======================================================================================
SCORE SUMMARY
=======================================================================================
Metric                          Cerebras (Qwen 3 235B)     Anthropic (Claude 3.5 Haiku)
---------------------------------------------------------------------------------------
Average Score                                 93.7 %                         89.1 %
Wins                                      90/100 (90%)                       6/100 (6%)
Winner                          Cerebras (Qwen 3 235B)

PERFORMANCE METRICS SUMMARY
=======================================================================================
Metric                            Cerebras (Qwen 3 235B)   Anthropic (Claude 3.5 Haiku)
---------------------------------------------------------------------------------------
TTFT (seconds)                 3.487 (P99: 10.21)     24.367 (P99: 34.23)
Input Throughput (tok/s)       24.27  (P99:  7.64)       3.45  (P99: 2.35)
Output Throughput (tok/s)    1086.97  (P99: 591.13)      69.12  (P99: 62.67)
Inter-Token Latency (ms)         1.0   (P99: 1.7)       14.5   (P99:16.0)

Note: Cerebras model is Qwen 3 235B Instruct (qwen-3-235b-a22b-instruct-2507)
Note: Anthropic model is Claude 3.5 Haiku (claude-3-5-haiku-20241022)
```

### Scores Explained

The scoring system evaluates generated code using a weighted rubric with four categories:

- **Correctness (0.30 weight)**: Does the code correctly implement the requirements?
  - Evaluates if the code solves the problem as specified
  - Checks for bugs, edge case handling, and proper error handling
  - Higher scores indicate code that works correctly and handles edge cases

- **Code Quality (0.30 weight)**: Is the code clean, readable, and well-structured?
  - Assesses code organization, readability, and maintainability
  - Evaluates use of proper abstractions, design patterns, and modularity
  - Higher scores indicate professional, maintainable code

- **Efficiency (0.20 weight)**: Is the code efficient in terms of time and space complexity?
  - Evaluates algorithm optimality and resource usage
  - Checks for unnecessary computation or memory waste
  - Higher scores indicate optimized, performant code

- **Documentation (0.20 weight)**: Is the code well-documented?
  - Assesses presence of docstrings, comments, and type hints
  - Evaluates how self-documenting the code is
  - Higher scores indicate code that is easy to understand and use

**Total Score**: The final score is calculated as a weighted sum of all four categories, with a maximum value of 1.000. Each category is scored from 0.0 to 1.0, then multiplied by its weight and summed. For example, a score of 0.938 means the code received high marks across all categories, with the weighted average being 93.8% of the maximum possible score.

### Performance Metrics Explained

The performance summary includes:

- **TTFT (Time To First Token)**: How long it takes to receive the first token of the response
  - Lower is better
  - Shows both average and P99 (99th percentile) to understand tail latency

- **Input Throughput**: Rate at which input tokens are processed (tokens/second)
  - Higher is better
  - Measures how efficiently the model processes the prompt

- **Output Throughput**: Rate at which output tokens are generated (tokens/second)
  - Higher is better
  - Key metric for generation speed

- **Inter-Token Latency**: Average time between consecutive output tokens (seconds)
  - Lower is better
  - Indicates generation smoothness and consistency

**P99 Percentile**: The 99th percentile shows the worst-case performance, helping identify tail latency issues. For example, if P99 TTFT is much higher than average, it indicates occasional slow responses.

### Interpretation

In the example above:
- **Cerebras (Qwen 3 235B)** shows significantly better performance across all metrics:
  - ~7x faster TTFT (3.5s vs 24.4s average) compared to Anthropic (Claude 3.5 Haiku)
  - ~15.7x higher output throughput (1087.0 vs 69.1 tokens/sec)
  - ~14.5x lower inter-token latency (1.0ms vs 14.5ms)
  - ~7x higher input throughput (24.3 vs 3.5 tokens/sec)
- **Code Quality**: Cerebras (Qwen 3 235B) achieved higher average scores (93.7% vs 89.1%) and won 90 out of 100 comparisons (90% win rate)
- **Consistency**: P99 values show Cerebras (Qwen 3 235B) maintains better performance even in worst-case scenarios, with P99 TTFT of 10.2s vs Anthropic's 34.2s

## License

MIT
