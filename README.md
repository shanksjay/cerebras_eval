# Cerebras CLI

A command-line interface for calling the Cerebras OpenAI-compatible chat API with built-in comparison and scoring capabilities.

## Features

- **Dual API Support**: Compare Cerebras (Qwen 3 235B Instruct) and Anthropic generation side-by-side
- **Automatic Scoring**: Code quality scoring using a comprehensive rubric
- **Performance Metrics**: Track TTFT, latency, and throughput
- **Batch Processing**: Process multiple prompts from JSON/JSONL files
- **Detailed Reports**: All results saved to files with comprehensive details

## Installation

```bash
pip install requests
```

## Setup

Set your API keys as environment variables:

```bash
export CEREBRAS_API_KEY="your-cerebras-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"  # Required for comparison and scoring
```

## Default Behavior

By default, the script:
- **Compares** Cerebras (Qwen 3 235B Instruct) vs Anthropic generation (`--compare` enabled)
- **Scores** all generated code (`--score` enabled)
- **Saves** all outputs to files (not printed to console)

To disable these defaults:
```bash
python cerebras_chat.py "prompt" --no-compare --no-score
```

## Usage

### Basic Usage (Single Prompt)

```bash
python cerebras_chat.py "Implement LRU cache in Python"
```

This will automatically:
- Generate code with both Cerebras (Qwen 3 235B Instruct) and Anthropic
- Score both outputs
- Display comparison tables
- Save detailed results to files

### With Custom Model and Temperature

```bash
python cerebras_chat.py "Write a function" --model qwen-3-235b-a22b-instruct-2507 --temperature 0.7
```

### With System Message

```bash
python cerebras_chat.py "Hello!" --system "You are a helpful coding assistant"
```

### Read from Stdin

```bash
echo "What is Python?" | python cerebras_chat.py
```

### Get Full JSON Response

```bash
python cerebras_chat.py "Hello!" --json
```

### Batch Processing from JSON/JSONL File

```bash
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
- `--compare`: Compare Cerebras (Qwen 3 235B Instruct) vs Anthropic generation (default: enabled)
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
===============================================================================================
📊 BATCH SUMMARY
===============================================================================================
Average Scores:
  Cerebras (Qwen 3 235B Instruct):  0.938
  Anthropic:                         0.862
  Winner: Cerebras (Qwen 3 235B Instruct)

Wins:
  Cerebras (Qwen 3 235B Instruct):  4/4
  Anthropic:                         0/4

===============================================================================================
⚡ PERFORMANCE METRICS SUMMARY
===============================================================================================
Metric                              Cerebras (Qwen 3 235B) Avg    Cerebras (Qwen 3 235B) P99    Anthropic Avg        Anthropic P99       
------------------------------------------------------------------------------------------------------------------------------------------------
TTFT (seconds)                      7.557                         11.816                       23.319               24.869              
Input Throughput (tok/s)            12.97                         24.08                        3.66                 4.08                
Output Throughput (tok/s)           581.57                        1110.30                       67.65                68.61               
Inter-Token Latency (s)             0.0023                        0.0039                        0.0148               0.0153              

Note: Cerebras model is Qwen 3 235B Instruct (qwen-3-235b-a22b-instruct-2507)
```

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
- **Cerebras (Qwen 3 235B Instruct)** shows significantly better performance across all metrics:
  - ~3x faster TTFT (7.6s vs 23.3s average)
  - ~8.6x higher output throughput (581.6 vs 67.7 tokens/sec)
  - ~6.4x lower inter-token latency (0.0023s vs 0.0148s)
- **Code Quality**: Cerebras (Qwen 3 235B Instruct) also achieved higher average scores (0.938 vs 0.862)
- **Consistency**: P99 values show Cerebras (Qwen 3 235B Instruct) maintains better performance even in worst-case scenarios

## License

MIT
