#!/usr/bin/env python3
"""
Cerebras Chat API Client

Calls the Cerebras OpenAI-compatible chat API with user prompts.
Can also compare Cerebras vs Anthropic generation with scoring.

Usage:
    # Basic usage with prompt as argument
    python cerebras_chat.py "Implement LRU cache in Python"
    
    # With custom model and temperature
    python cerebras_chat.py "Write a function" --model llama3.1-8b --temperature 0.7
    
    # With multiple messages (system + user)
    python cerebras_chat.py "Hello!" --system "You are a helpful coding assistant"
    
    # Read prompt from stdin
    echo "What is Python?" | python cerebras_chat.py
    
    # Generate code and score it
    python cerebras_chat.py "Implement LRU cache in Python" --score
    
    # Compare Cerebras vs Anthropic generation
    python cerebras_chat.py "Implement LRU cache in Python" --compare
    
Environment Variables:
    CEREBRAS_API_KEY: Required for Cerebras API.
    ANTHROPIC_API_KEY: Required for Anthropic API (when using --compare).
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from typing import Optional, List, Dict, Any, Tuple

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' library not found. Install with: pip install anthropic")
    sys.exit(1)


def call_cerebras_api(
    prompt: str,
    api_key: str,
    model: str = "llama3.1-8b",
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    max_completion_tokens: int = -1,
    top_p: float = 1.0,
    seed: Optional[int] = 0,
    stream: bool = False,
    track_timing: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """
    Call Cerebras OpenAI-compatible chat API.
    
    Args:
        prompt: User prompt message
        api_key: Cerebras API key
        model: Model name (default: llama3.1-8b)
        system_message: Optional system message
        temperature: Sampling temperature (0.0 = deterministic)
        max_completion_tokens: Max tokens to generate (-1 = no limit)
        top_p: Nucleus sampling parameter
        seed: Random seed for reproducibility
        stream: Whether to stream responses
        track_timing: Whether to track performance metrics
        
    Returns:
        Tuple of (API response dictionary, performance metrics dictionary or None)
    """
    url = "https://api.cerebras.ai/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "seed": seed,
        "stream": stream,
    }
    
    performance_metrics = None
    
    try:
        # Track timing
        if track_timing:
            request_start = time.time()
            # For non-streaming, we approximate TTFT as the full request time
            # In a real streaming implementation, we'd track when first token arrives
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        if track_timing:
            request_end = time.time()
            total_time = request_end - request_start
            
            # Extract token usage from response
            response_data = response.json()
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            
            # Calculate performance metrics
            # For non-streaming, TTFT is approximately the full latency
            ttft = total_time
            
            # Latency between tokens (average time per output token)
            if output_tokens > 0:
                token_latency = total_time / output_tokens
            else:
                token_latency = 0.0
            
            # Input throughput (tokens per second)
            if total_time > 0:
                input_throughput = input_tokens / total_time
            else:
                input_throughput = 0.0
            
            # Output throughput (tokens per second)
            if total_time > 0 and output_tokens > 0:
                output_throughput = output_tokens / total_time
            else:
                output_throughput = 0.0
            
            performance_metrics = {
                "ttft_seconds": ttft,
                "token_latency_seconds": token_latency,
                "input_throughput_tokens_per_sec": input_throughput,
                "output_throughput_tokens_per_sec": output_throughput,
                "total_time_seconds": total_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            
            return response_data, performance_metrics
        
        return response.json(), None
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Cerebras API: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"API Error Details: {error_detail}", file=sys.stderr)
            except:
                print(f"Response: {e.response.text}", file=sys.stderr)
        raise


def extract_content(response: Dict[str, Any]) -> str:
    """Extract content from API response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Error extracting content from response: {e}", file=sys.stderr)
        print(f"Response structure: {response}", file=sys.stderr)
        raise


def call_anthropic_api(
    prompt: str,
    api_key: str,
    model: str = "claude-3-5-haiku-20241022",
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    track_timing: bool = False,
) -> Tuple[Any, Optional[Dict[str, float]]]:
    """
    Call Anthropic Claude API for chat completions using anthropic library.
    
    Args:
        prompt: User prompt message
        api_key: Anthropic API key
        model: Model name (default: claude-3-5-haiku-20241022)
        system_message: Optional system message
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Max tokens to generate
        track_timing: Whether to track performance metrics
        
    Returns:
        Tuple of (API response object, performance metrics dictionary or None)
    """
    performance_metrics = None
    
    try:
        # Create client with API key
        client = anthropic.Anthropic(api_key=api_key)
        
        # Track timing
        if track_timing:
            request_start = time.time()
        
        # Build messages list
        messages = [{"role": "user", "content": prompt}]
        
        # Build API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        if system_message:
            api_params["system"] = system_message
        
        if temperature > 0:
            api_params["temperature"] = temperature
        
        # Generate content
        response = client.messages.create(**api_params)
        
        if track_timing:
            request_end = time.time()
            total_time = request_end - request_start
            
            # Extract token usage from response
            try:
                if hasattr(response, 'usage'):
                    usage = response.usage
                    input_tokens = getattr(usage, 'input_tokens', 0)
                    output_tokens = getattr(usage, 'output_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                else:
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
            except Exception:
                # If we can't extract token usage, set to 0
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
            
            # Calculate performance metrics
            ttft = total_time  # For non-streaming, TTFT is approximately the full latency
            
            if output_tokens > 0:
                token_latency = total_time / output_tokens
            else:
                token_latency = 0.0
            
            if total_time > 0:
                input_throughput = input_tokens / total_time
            else:
                input_throughput = 0.0
            
            if total_time > 0 and output_tokens > 0:
                output_throughput = output_tokens / total_time
            else:
                output_throughput = 0.0
            
            performance_metrics = {
                "ttft_seconds": ttft,
                "token_latency_seconds": token_latency,
                "input_throughput_tokens_per_sec": input_throughput,
                "output_throughput_tokens_per_sec": output_throughput,
                "total_time_seconds": total_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            
            return response, performance_metrics
        
        return response, None
        
    except Exception as e:
        print(f"Error calling Anthropic API: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise


def extract_anthropic_content(response: Any) -> str:
    """Extract content from Anthropic Claude API response object."""
    try:
        # Anthropic API returns content in response.content
        if hasattr(response, 'content'):
            # content is a list of content blocks
            if isinstance(response.content, list) and len(response.content) > 0:
                # Get the first content block
                first_block = response.content[0]
                if hasattr(first_block, 'text'):
                    return first_block.text
                elif isinstance(first_block, dict) and 'text' in first_block:
                    return first_block['text']
        # Fallback: try to access as dict
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
            if isinstance(content, list) and len(content) > 0:
                first_block = content[0]
                if isinstance(first_block, dict) and 'text' in first_block:
                    return first_block['text']
        return ""
    except (AttributeError, IndexError, KeyError) as e:
        print(f"Error extracting content from Anthropic response: {e}", file=sys.stderr)
        print(f"Response type: {type(response)}", file=sys.stderr)
        if hasattr(response, '__dict__'):
            print(f"Response attributes: {dir(response)}", file=sys.stderr)
        raise


def score_code(
    code: str,
    prompt: str,
    api_key: str,
    model: str = "claude-3-5-haiku-20241022",
    language: str = "python",
    track_timing: bool = False,
    use_anthropic: bool = True,
) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """
    Score code using a rubric-based evaluation.
    
    Args:
        code: The generated code to score
        prompt: The original prompt that generated the code
        api_key: API key (Anthropic or Cerebras)
        model: Model name for scoring (default: claude-3-5-haiku-20241022)
        language: Programming language (default: python)
        track_timing: Whether to track performance metrics
        use_anthropic: Whether to use Anthropic API (True) or Cerebras API (False)
        
    Returns:
        Tuple of (API response with scoring results, performance metrics or None)
    """
    scoring_rubric = """You are a strict scoring function for code quality evaluation.

SCORING RUBRIC:
1. Correctness (0.30): Does the code correctly implement the requirements? Are there bugs, edge cases handled, proper error handling?
2. Code Quality (0.30): Is the code clean, readable, well-structured? Are there proper abstractions, design patterns, modularity?
3. Efficiency (0.20): Is the code efficient? Are algorithms optimal? Is there unnecessary computation or memory usage?
4. Documentation (0.20): Are there docstrings, comments, type hints? Is the code self-documenting?

Provide a score from 0.0 to 1.0 for each category, then calculate the weighted total.
Format your response as JSON with the following structure:
{
    "correctness": 0.0-1.0,
    "code_quality": 0.0-1.0,
    "efficiency": 0.0-1.0,
    "documentation": 0.0-1.0,
    "total_score": 0.0-1.0,
    "feedback": "Brief explanation of scores"
}"""

    scoring_prompt = f"""Code to evaluate:
```{language}
{code}
```

Original prompt:
{prompt[:200]}{"..." if len(prompt) > 200 else ""}

Evaluate this code according to the rubric and provide scores in JSON format."""

    if use_anthropic:
        return call_anthropic_api(
            prompt=scoring_prompt,
            api_key=api_key,
            model=model,
            system_message=scoring_rubric,
            temperature=0.0,  # Deterministic scoring
            max_tokens=500,
            track_timing=track_timing,
        )
    else:
        return call_cerebras_api(
            prompt=scoring_prompt,
            api_key=api_key,
            model=model,
            system_message=scoring_rubric,
            temperature=0.0,  # Deterministic scoring
            max_completion_tokens=500,
            track_timing=track_timing,
        )


def clean_json_response(raw_text: str) -> str:
    """
    Clean JSON response by removing markdown code blocks.
    
    Args:
        raw_text: Raw text response that may contain markdown code blocks
        
    Returns:
        Cleaned text with markdown code blocks removed
    """
    # Remove markdown code blocks (```json or ```)
    clean_text = re.sub(r'```(?:json)?\n?|```', '', raw_text).strip()
    return clean_text


def parse_score_response(response: Any, is_anthropic: bool = True) -> Dict[str, float]:
    """
    Parse scoring response and extract scores.
    
    Args:
        response: API response from scoring call (dict for Cerebras, object for Anthropic)
        is_anthropic: Whether the response is from Anthropic API
        
    Returns:
        Dictionary with scores and feedback
    """
    try:
        if is_anthropic:
            content = extract_anthropic_content(response)
        else:
            content = extract_content(response)
        
        # Clean the content to remove markdown code blocks
        cleaned_content = clean_json_response(content)
        
        # Try to extract JSON from the response
        # Look for JSON object in the response (handle multiline JSON)
        json_match = re.search(r'\{.*?"correctness".*?\}', cleaned_content, re.DOTALL)
        if json_match:
            try:
                score_data = json.loads(json_match.group())
                return {
                    "correctness": float(score_data.get("correctness", 0.0)),
                    "code_quality": float(score_data.get("code_quality", 0.0)),
                    "efficiency": float(score_data.get("efficiency", 0.0)),
                    "documentation": float(score_data.get("documentation", 0.0)),
                    "total_score": float(score_data.get("total_score", 0.0)),
                    "feedback": score_data.get("feedback", "No feedback provided"),
                }
            except json.JSONDecodeError:
                # If the matched JSON is incomplete, try parsing the cleaned content directly
                pass
        
        # Fallback: try to parse the entire cleaned content as JSON
        try:
            score_data = json.loads(cleaned_content)
            return {
                "correctness": float(score_data.get("correctness", 0.0)),
                "code_quality": float(score_data.get("code_quality", 0.0)),
                "efficiency": float(score_data.get("efficiency", 0.0)),
                "documentation": float(score_data.get("documentation", 0.0)),
                "total_score": float(score_data.get("total_score", 0.0)),
                "feedback": score_data.get("feedback", "No feedback provided"),
            }
        except json.JSONDecodeError:
            # Try to find and parse just the JSON part more aggressively
            # Look for JSON that might span multiple lines
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, cleaned_content, re.DOTALL)
            for match in json_matches:
                try:
                    score_data = json.loads(match)
                    if "correctness" in score_data:
                        return {
                            "correctness": float(score_data.get("correctness", 0.0)),
                            "code_quality": float(score_data.get("code_quality", 0.0)),
                            "efficiency": float(score_data.get("efficiency", 0.0)),
                            "documentation": float(score_data.get("documentation", 0.0)),
                            "total_score": float(score_data.get("total_score", 0.0)),
                            "feedback": score_data.get("feedback", "No feedback provided"),
                        }
                except json.JSONDecodeError:
                    continue
            
            # If all parsing attempts fail, raise the error
            raise json.JSONDecodeError("Could not parse JSON from response", cleaned_content, 0)
            
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Could not parse score response: {e}", file=sys.stderr)
        print(f"Response content: {content[:500] if 'content' in locals() else 'N/A'}", file=sys.stderr)
        return {
            "correctness": 0.0,
            "code_quality": 0.0,
            "efficiency": 0.0,
            "documentation": 0.0,
            "total_score": 0.0,
            "feedback": "Error parsing score response",
        }


def process_single_prompt(prompt: str, language: str, args, output_dir: str, prompt_id: int) -> Dict[str, Any]:
    """
    Process a single prompt through the comparison pipeline.
    
    Returns a dictionary with results for summary generation.
    """
    # Get both API keys
    cerebras_key = args.api_key or os.getenv("CEREBRAS_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not cerebras_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set and --api-key not provided")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    # Generate with Cerebras
    cerebras_response, cerebras_perf = call_cerebras_api(
        prompt=prompt,
        api_key=cerebras_key,
        model=args.model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        seed=args.seed,
        stream=args.stream,
        track_timing=True,
    )
    cerebras_content = extract_content(cerebras_response)
    
    # Generate with Anthropic
    anthropic_response, anthropic_perf = call_anthropic_api(
        prompt=prompt,
        api_key=anthropic_key,
        model=args.anthropic_model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens if args.max_completion_tokens > 0 else 4096,
        track_timing=True,
    )
    anthropic_content = extract_anthropic_content(anthropic_response)
    
    # Score both outputs using Anthropic
    cerebras_score_response, cerebras_score_perf = score_code(
        code=cerebras_content,
        prompt=prompt,
        api_key=anthropic_key,
        model=args.anthropic_model,
        language=language,
        track_timing=True,
        use_anthropic=True,
    )
    cerebras_scores = parse_score_response(cerebras_score_response, is_anthropic=True)
    
    anthropic_score_response, anthropic_score_perf = score_code(
        code=anthropic_content,
        prompt=prompt,
        api_key=anthropic_key,
        model=args.anthropic_model,
        language=language,
        track_timing=True,
        use_anthropic=True,
    )
    anthropic_scores = parse_score_response(anthropic_score_response, is_anthropic=True)
    
    # Save outputs to files
    file_prefix = f"prompt_{prompt_id:03d}"
    
    cerebras_file = os.path.join(output_dir, f"{file_prefix}_cerebras.txt")
    anthropic_file = os.path.join(output_dir, f"{file_prefix}_anthropic.txt")
    
    # Save Cerebras output with all details
    with open(cerebras_file, 'w', encoding='utf-8') as f:
        f.write("="*95 + "\n")
        f.write("CEREBRAS CODE GENERATION OUTPUT\n")
        f.write("="*95 + "\n\n")
        f.write(f"Prompt ID: {prompt_id}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*95 + "\n")
        if cerebras_perf:
            f.write(f"Time To First Token (TTFT):     {cerebras_perf['ttft_seconds']*1000:.2f} ms\n")
            f.write(f"Latency Between Tokens:        {cerebras_perf['token_latency_seconds']*1000:.2f} ms/token\n")
            f.write(f"Input Throughput:               {cerebras_perf['input_throughput_tokens_per_sec']:.2f} tokens/sec\n")
            f.write(f"Output Throughput:              {cerebras_perf['output_throughput_tokens_per_sec']:.2f} tokens/sec\n")
            f.write(f"Total Time:                     {cerebras_perf['total_time_seconds']:.3f} seconds\n")
            f.write(f"Input Tokens:                   {cerebras_perf['input_tokens']}\n")
            f.write(f"Output Tokens:                  {cerebras_perf['output_tokens']}\n")
            f.write(f"Total Tokens:                   {cerebras_perf['total_tokens']}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("CODE QUALITY SCORES\n")
        f.write("="*95 + "\n")
        f.write(f"Correctness (0.30):  {cerebras_scores['correctness']:.3f} × 0.30 = {cerebras_scores['correctness'] * 0.30:.3f}\n")
        f.write(f"Code Quality (0.30): {cerebras_scores['code_quality']:.3f} × 0.30 = {cerebras_scores['code_quality'] * 0.30:.3f}\n")
        f.write(f"Efficiency (0.20):   {cerebras_scores['efficiency']:.3f} × 0.20 = {cerebras_scores['efficiency'] * 0.20:.3f}\n")
        f.write(f"Documentation (0.20): {cerebras_scores['documentation']:.3f} × 0.20 = {cerebras_scores['documentation'] * 0.20:.3f}\n")
        f.write(f"\nTOTAL SCORE: {cerebras_scores['total_score']:.3f} / 1.000\n")
        f.write(f"\nFeedback: {cerebras_scores['feedback']}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("GENERATED CODE\n")
        f.write("="*95 + "\n\n")
        f.write(cerebras_content)
    
    # Save Anthropic output with all details
    with open(anthropic_file, 'w', encoding='utf-8') as f:
        f.write("="*95 + "\n")
        f.write("ANTHROPIC CODE GENERATION OUTPUT\n")
        f.write("="*95 + "\n\n")
        f.write(f"Prompt ID: {prompt_id}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Model: {args.anthropic_model}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*95 + "\n")
        if anthropic_perf:
            f.write(f"Time To First Token (TTFT):     {anthropic_perf['ttft_seconds']*1000:.2f} ms\n")
            f.write(f"Latency Between Tokens:        {anthropic_perf['token_latency_seconds']*1000:.2f} ms/token\n")
            f.write(f"Input Throughput:               {anthropic_perf['input_throughput_tokens_per_sec']:.2f} tokens/sec\n")
            f.write(f"Output Throughput:              {anthropic_perf['output_throughput_tokens_per_sec']:.2f} tokens/sec\n")
            f.write(f"Total Time:                     {anthropic_perf['total_time_seconds']:.3f} seconds\n")
            f.write(f"Input Tokens:                   {anthropic_perf['input_tokens']}\n")
            f.write(f"Output Tokens:                  {anthropic_perf['output_tokens']}\n")
            f.write(f"Total Tokens:                   {anthropic_perf['total_tokens']}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("CODE QUALITY SCORES\n")
        f.write("="*95 + "\n")
        f.write(f"Correctness (0.30):  {anthropic_scores['correctness']:.3f} × 0.30 = {anthropic_scores['correctness'] * 0.30:.3f}\n")
        f.write(f"Code Quality (0.30): {anthropic_scores['code_quality']:.3f} × 0.30 = {anthropic_scores['code_quality'] * 0.30:.3f}\n")
        f.write(f"Efficiency (0.20):   {anthropic_scores['efficiency']:.3f} × 0.20 = {anthropic_scores['efficiency'] * 0.20:.3f}\n")
        f.write(f"Documentation (0.20): {anthropic_scores['documentation']:.3f} × 0.20 = {anthropic_scores['documentation'] * 0.20:.3f}\n")
        f.write(f"\nTOTAL SCORE: {anthropic_scores['total_score']:.3f} / 1.000\n")
        f.write(f"\nFeedback: {anthropic_scores['feedback']}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("GENERATED CODE\n")
        f.write("="*95 + "\n\n")
        f.write(anthropic_content)
    
    # Return result for summary
    return {
        'prompt_id': prompt_id,
        'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
        'language': language,
        'cerebras': {
            'score': cerebras_scores['total_score'],
            'performance': cerebras_perf,
            'file': cerebras_file,
        },
        'anthropic': {
            'score': anthropic_scores['total_score'],
            'performance': anthropic_perf,
            'file': anthropic_file,
        },
        'success': True,
    }


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate the percentile of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def generate_batch_summary(results: List[Dict[str, Any]], output_dir: str, timestamp: str):
    """Generate a comprehensive summary of batch processing results."""
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    
    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*95 + "\n")
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("="*95 + "\n\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total Prompts: {len(results)}\n")
        f.write(f"Successful: {len(successful_results)}\n")
        f.write(f"Failed: {len(failed_results)}\n")
        f.write("\n" + "="*95 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*95 + "\n\n")
        
        if successful_results:
            # Calculate averages
            cerebras_scores = [r['cerebras']['score'] for r in successful_results]
            anthropic_scores = [r['anthropic']['score'] for r in successful_results]
            
            cerebras_avg_score = sum(cerebras_scores) / len(cerebras_scores)
            anthropic_avg_score = sum(anthropic_scores) / len(anthropic_scores)
            
            # Performance averages
            cerebras_total_times = [r['cerebras']['performance']['total_time_seconds'] for r in successful_results if r['cerebras']['performance']]
            anthropic_total_times = [r['anthropic']['performance']['total_time_seconds'] for r in successful_results if r['anthropic']['performance']]
            
            cerebras_avg_time = sum(cerebras_total_times) / len(cerebras_total_times) if cerebras_total_times else 0
            anthropic_avg_time = sum(anthropic_total_times) / len(anthropic_total_times) if anthropic_total_times else 0
            
            cerebras_output_throughputs = [r['cerebras']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['cerebras']['performance']]
            anthropic_output_throughputs = [r['anthropic']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['anthropic']['performance']]
            
            cerebras_avg_throughput = sum(cerebras_output_throughputs) / len(cerebras_output_throughputs) if cerebras_output_throughputs else 0
            anthropic_avg_throughput = sum(anthropic_output_throughputs) / len(anthropic_output_throughputs) if anthropic_output_throughputs else 0
            
            f.write("AVERAGE CODE QUALITY SCORES:\n")
            f.write(f"  Cerebras:  {cerebras_avg_score:.3f} / 1.000\n")
            f.write(f"  Anthropic: {anthropic_avg_score:.3f} / 1.000\n")
            f.write(f"  Winner: {'Cerebras' if cerebras_avg_score > anthropic_avg_score else 'Anthropic' if anthropic_avg_score > cerebras_avg_score else 'Tie'}\n")
            f.write("\n")
            
            # Collect performance metrics for detailed analysis
            cerebras_ttft = [r['cerebras']['performance']['ttft_seconds'] for r in successful_results if r['cerebras']['performance']]
            anthropic_ttft = [r['anthropic']['performance']['ttft_seconds'] for r in successful_results if r['anthropic']['performance']]
            
            cerebras_input_throughput = [r['cerebras']['performance']['input_throughput_tokens_per_sec'] for r in successful_results if r['cerebras']['performance']]
            anthropic_input_throughput = [r['anthropic']['performance']['input_throughput_tokens_per_sec'] for r in successful_results if r['anthropic']['performance']]
            
            cerebras_output_throughput = [r['cerebras']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['cerebras']['performance']]
            anthropic_output_throughput = [r['anthropic']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['anthropic']['performance']]
            
            cerebras_token_latency = [r['cerebras']['performance']['token_latency_seconds'] for r in successful_results if r['cerebras']['performance']]
            anthropic_token_latency = [r['anthropic']['performance']['token_latency_seconds'] for r in successful_results if r['anthropic']['performance']]
            
            # Calculate averages and P99
            def calc_stats(values):
                if not values:
                    return 0.0, 0.0
                avg = sum(values) / len(values)
                p99 = calculate_percentile(values, 99.0)
                return avg, p99
            
            c_ttft_avg, c_ttft_p99 = calc_stats(cerebras_ttft)
        g_ttft_avg, g_ttft_p99 = calc_stats(anthropic_ttft)
        
        c_input_tp_avg, c_input_tp_p99 = calc_stats(cerebras_input_throughput)
        g_input_tp_avg, g_input_tp_p99 = calc_stats(anthropic_input_throughput)
        
        c_output_tp_avg, c_output_tp_p99 = calc_stats(cerebras_output_throughput)
        g_output_tp_avg, g_output_tp_p99 = calc_stats(anthropic_output_throughput)
        
        c_latency_avg, c_latency_p99 = calc_stats(cerebras_token_latency)
        g_latency_avg, g_latency_p99 = calc_stats(anthropic_token_latency)
        
        f.write("PERFORMANCE METRICS SUMMARY:\n")
        f.write("="*95 + "\n")
        f.write(f"{'Metric':<35} {'Cerebras Avg':<20} {'Cerebras P99':<20} {'Anthropic Avg':<20} {'Anthropic P99':<20}\n")
        f.write("-"*95 + "\n")
        f.write(f"{'TTFT (seconds)':<35} {c_ttft_avg:<20.3f} {c_ttft_p99:<20.3f} {g_ttft_avg:<20.3f} {g_ttft_p99:<20.3f}\n")
        f.write(f"{'Input Throughput (tok/s)':<35} {c_input_tp_avg:<20.2f} {c_input_tp_p99:<20.2f} {g_input_tp_avg:<20.2f} {g_input_tp_p99:<20.2f}\n")
        f.write(f"{'Output Throughput (tok/s)':<35} {c_output_tp_avg:<20.2f} {c_output_tp_p99:<20.2f} {g_output_tp_avg:<20.2f} {g_output_tp_p99:<20.2f}\n")
        f.write(f"{'Inter-Token Latency (s)':<35} {c_latency_avg:<20.4f} {c_latency_p99:<20.4f} {g_latency_avg:<20.4f} {g_latency_p99:<20.4f}\n")
        f.write("\n")
        
        f.write("LEGACY PERFORMANCE METRICS (for reference):\n")
        f.write(f"  Cerebras Average Time:  {cerebras_avg_time:.3f} seconds\n")
        f.write(f"  Anthropic Average Time: {anthropic_avg_time:.3f} seconds\n")
        f.write(f"  Speed Up: {anthropic_avg_time / cerebras_avg_time:.2f}x (Cerebras faster)\n" if cerebras_avg_time > 0 else "  Speed Up: N/A\n")
        f.write(f"  Cerebras Average Throughput:  {cerebras_avg_throughput:.2f} tokens/sec\n")
        f.write(f"  Anthropic Average Throughput: {anthropic_avg_throughput:.2f} tokens/sec\n")
        f.write("\n")
        
        # Count winners
        cerebras_wins_score = sum(1 for r in successful_results if r['cerebras']['score'] > r['anthropic']['score'])
        anthropic_wins_score = sum(1 for r in successful_results if r['anthropic']['score'] > r['cerebras']['score'])
        ties_score = len(successful_results) - cerebras_wins_score - anthropic_wins_score
        
        f.write("SCORE COMPARISON BREAKDOWN:\n")
        f.write(f"  Cerebras Wins:  {cerebras_wins_score} ({100*cerebras_wins_score/len(successful_results):.1f}%)\n")
        f.write(f"  Anthropic Wins: {anthropic_wins_score} ({100*anthropic_wins_score/len(successful_results):.1f}%)\n")
        f.write(f"  Ties:           {ties_score} ({100*ties_score/len(successful_results):.1f}%)\n")
        f.write("\n")
        
        f.write("="*95 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*95 + "\n\n")
        
        for result in results:
            f.write(f"Prompt ID: {result.get('prompt_id', result.get('line_num', 'N/A'))}\n")
            f.write(f"Language: {result.get('language', 'N/A')}\n")
            f.write(f"Prompt: {result.get('prompt', 'N/A')}\n")
            
            if result.get('success'):
                f.write(f"  Cerebras Score:  {result['cerebras']['score']:.3f}\n")
                f.write(f"  Anthropic Score: {result['anthropic']['score']:.3f}\n")
                f.write(f"  Winner: {'Cerebras' if result['cerebras']['score'] > result['anthropic']['score'] else 'Anthropic' if result['anthropic']['score'] > result['cerebras']['score'] else 'Tie'}\n")
                if result['cerebras']['performance']:
                    f.write(f"  Cerebras Time:  {result['cerebras']['performance']['total_time_seconds']:.3f}s\n")
                if result['anthropic']['performance']:
                    f.write(f"  Anthropic Time: {result['anthropic']['performance']['total_time_seconds']:.3f}s\n")
                f.write(f"  Files: {os.path.basename(result['cerebras']['file'])}, {os.path.basename(result['anthropic']['file'])}\n")
            else:
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
    
    # Also print summary to stderr
    print(f"\n{'='*95}", file=sys.stderr)
    print("📊 BATCH SUMMARY", file=sys.stderr)
    print(f"{'='*95}", file=sys.stderr)
    if successful_results:
        cerebras_scores = [r['cerebras']['score'] for r in successful_results]
        anthropic_scores = [r['anthropic']['score'] for r in successful_results]
        cerebras_avg = sum(cerebras_scores) / len(cerebras_scores)
        anthropic_avg = sum(anthropic_scores) / len(anthropic_scores)
        
        print(f"Average Scores:", file=sys.stderr)
        print(f"  Cerebras:  {cerebras_avg:.3f}", file=sys.stderr)
        print(f"  Anthropic: {anthropic_avg:.3f}", file=sys.stderr)
        print(f"  Winner: {'Cerebras' if cerebras_avg > anthropic_avg else 'Anthropic' if anthropic_avg > cerebras_avg else 'Tie'}", file=sys.stderr)
        
        cerebras_wins = sum(1 for r in successful_results if r['cerebras']['score'] > r['anthropic']['score'])
        anthropic_wins = sum(1 for r in successful_results if r['anthropic']['score'] > r['cerebras']['score'])
        print(f"\nWins:", file=sys.stderr)
        print(f"  Cerebras:  {cerebras_wins}/{len(successful_results)}", file=sys.stderr)
        print(f"  Anthropic: {anthropic_wins}/{len(successful_results)}", file=sys.stderr)
        
        # Performance metrics summary
        cerebras_ttft = [r['cerebras']['performance']['ttft_seconds'] for r in successful_results if r['cerebras']['performance']]
        anthropic_ttft = [r['anthropic']['performance']['ttft_seconds'] for r in successful_results if r['anthropic']['performance']]
        
        cerebras_input_throughput = [r['cerebras']['performance']['input_throughput_tokens_per_sec'] for r in successful_results if r['cerebras']['performance']]
        anthropic_input_throughput = [r['anthropic']['performance']['input_throughput_tokens_per_sec'] for r in successful_results if r['anthropic']['performance']]
        
        cerebras_output_throughput = [r['cerebras']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['cerebras']['performance']]
        anthropic_output_throughput = [r['anthropic']['performance']['output_throughput_tokens_per_sec'] for r in successful_results if r['anthropic']['performance']]
        
        cerebras_token_latency = [r['cerebras']['performance']['token_latency_seconds'] for r in successful_results if r['cerebras']['performance']]
        anthropic_token_latency = [r['anthropic']['performance']['token_latency_seconds'] for r in successful_results if r['anthropic']['performance']]
        
        def calc_stats(values):
            if not values:
                return 0.0, 0.0
            avg = sum(values) / len(values)
            p99 = calculate_percentile(values, 99.0)
            return avg, p99
        
        c_ttft_avg, c_ttft_p99 = calc_stats(cerebras_ttft)
        g_ttft_avg, g_ttft_p99 = calc_stats(anthropic_ttft)
        
        c_input_tp_avg, c_input_tp_p99 = calc_stats(cerebras_input_throughput)
        g_input_tp_avg, g_input_tp_p99 = calc_stats(anthropic_input_throughput)
        
        c_output_tp_avg, c_output_tp_p99 = calc_stats(cerebras_output_throughput)
        g_output_tp_avg, g_output_tp_p99 = calc_stats(anthropic_output_throughput)
        
        c_latency_avg, c_latency_p99 = calc_stats(cerebras_token_latency)
        g_latency_avg, g_latency_p99 = calc_stats(anthropic_token_latency)
        
        print(f"\n{'='*95}", file=sys.stderr)
        print("⚡ PERFORMANCE METRICS SUMMARY", file=sys.stderr)
        print(f"{'='*95}", file=sys.stderr)
        print(f"{'Metric':<35} {'Cerebras Avg':<20} {'Cerebras P99':<20} {'Anthropic Avg':<20} {'Anthropic P99':<20}", file=sys.stderr)
        print("-"*95, file=sys.stderr)
        print(f"{'TTFT (seconds)':<35} {c_ttft_avg:<20.3f} {c_ttft_p99:<20.3f} {g_ttft_avg:<20.3f} {g_ttft_p99:<20.3f}", file=sys.stderr)
        print(f"{'Input Throughput (tok/s)':<35} {c_input_tp_avg:<20.2f} {c_input_tp_p99:<20.2f} {g_input_tp_avg:<20.2f} {g_input_tp_p99:<20.2f}", file=sys.stderr)
        print(f"{'Output Throughput (tok/s)':<35} {c_output_tp_avg:<20.2f} {c_output_tp_p99:<20.2f} {g_output_tp_avg:<20.2f} {g_output_tp_p99:<20.2f}", file=sys.stderr)
        print(f"{'Inter-Token Latency (s)':<35} {c_latency_avg:<20.4f} {c_latency_p99:<20.4f} {g_latency_avg:<20.4f} {g_latency_p99:<20.4f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Call Cerebras OpenAI-compatible chat API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="User prompt (if not provided, reads from stdin or --input-file)"
    )
    
    parser.add_argument(
        "--input-file",
        dest="input_file",
        help="JSON/JSONL file with list of prompts. Each line should be JSON with 'prompt' and optional 'language' fields."
    )
    
    parser.add_argument(
        "--model",
        default="qwen-3-235b-a22b-instruct-2507",
        help="Model name (default: qwen-3-235b-a22b-instruct-2507)"
    )
    
    parser.add_argument(
        "--system",
        dest="system_message",
        help="System message to set assistant behavior"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = deterministic, default: 0.0)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        dest="max_completion_tokens",
        help="Maximum completion tokens (-1 = no limit, default: -1)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter (default: 1.0)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream responses (not yet implemented)"
    )
    
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Cerebras API key (default: from CEREBRAS_API_KEY env var)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON response instead of just content"
    )
    
    parser.add_argument(
        "--no-score",
        dest="score",
        action="store_false",
        default=True,
        help="Disable scoring of generated code (scoring is enabled by default)"
    )
    
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score the generated code using a rubric (Correctness 0.30, Code Quality 0.30, Efficiency 0.20, Documentation 0.20) (default: enabled)"
    )
    
    parser.add_argument(
        "--language",
        default="python",
        help="Programming language for scoring (default: python)"
    )
    
    parser.add_argument(
        "--score-model",
        dest="score_model",
        default=None,
        help="Model to use for scoring (default: Anthropic claude-3-5-haiku-20241022)"
    )
    
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        default=True,
        help="Disable comparison mode (use single API only) (comparison is enabled by default)"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Cerebras vs Anthropic generation (requires both API keys) (default: enabled)"
    )
    
    parser.add_argument(
        "--anthropic-model",
        dest="anthropic_model",
        default="claude-3-5-haiku-20241022",
        help="Anthropic model to use (default: claude-3-5-haiku-20241022)"
    )
    
    args = parser.parse_args()
    
    # Handle input file mode
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        # Read prompts from file
        prompts_data = []
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if 'prompt' not in data:
                            print(f"Warning: Line {line_num} missing 'prompt' field, skipping", file=sys.stderr)
                            continue
                        prompts_data.append({
                            'prompt': data['prompt'],
                            'language': data.get('language', args.language),
                            'line_num': line_num
                        })
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {line_num} is not valid JSON, skipping: {e}", file=sys.stderr)
                        continue
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
        
        if not prompts_data:
            print("Error: No valid prompts found in input file", file=sys.stderr)
            sys.exit(1)
        
        print(f"📋 Loaded {len(prompts_data)} prompts from {args.input_file}", file=sys.stderr)
        
        # Process each prompt
        results = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure all batch outputs are saved under the output directory
        # Get absolute path to output directory to ensure consistency
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base_dir = os.path.join(script_dir, "output")
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Create batch output directory under output/
        output_dir = os.path.join(output_base_dir, f"batch_output_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, prompt_data in enumerate(prompts_data, 1):
            prompt = prompt_data['prompt']
            language = prompt_data['language']
            line_num = prompt_data['line_num']
            
            print(f"\n{'='*95}", file=sys.stderr)
            print(f"Processing prompt {idx}/{len(prompts_data)} (line {line_num})", file=sys.stderr)
            print(f"{'='*95}", file=sys.stderr)
            
            try:
                # Process this prompt using the comparison logic
                result = process_single_prompt(
                    prompt=prompt,
                    language=language,
                    args=args,
                    output_dir=output_dir,
                    prompt_id=idx
                )
                result['line_num'] = line_num
                results.append(result)
                print(f"✓ Completed prompt {idx}/{len(prompts_data)}", file=sys.stderr)
            except Exception as e:
                print(f"✗ Error processing prompt {idx}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                results.append({
                    'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    'language': language,
                    'line_num': line_num,
                    'error': str(e),
                    'success': False
                })
        
        # Generate summary
        generate_batch_summary(results, output_dir, timestamp)
        
        print(f"\n{'='*95}", file=sys.stderr)
        print(f"✅ BATCH PROCESSING COMPLETE", file=sys.stderr)
        print(f"{'='*95}", file=sys.stderr)
        print(f"📁 All outputs saved to: {output_dir}/", file=sys.stderr)
        print(f"📊 Summary saved to: {output_dir}/summary_{timestamp}.txt", file=sys.stderr)
        
        sys.exit(0)
    
    # Get prompt (single prompt mode)
    if args.prompt:
        prompt = args.prompt
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Error: No prompt provided and stdin is empty", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: Empty prompt", file=sys.stderr)
            sys.exit(1)
    
    # Handle comparison mode
    if args.compare:
        # Get both API keys
        cerebras_key = args.api_key or os.getenv("CEREBRAS_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not cerebras_key:
            print("Error: CEREBRAS_API_KEY environment variable not set and --api-key not provided", file=sys.stderr)
            sys.exit(1)
        if not anthropic_key:
            print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        
        try:
            print("="*60, file=sys.stderr)
            print("🔄 COMPARING CEREBRAS vs ANTHROPIC", file=sys.stderr)
            print("="*60, file=sys.stderr)
            
            # Generate with Cerebras
            print("\n[1/4] Generating with Cerebras...", file=sys.stderr)
            cerebras_response, cerebras_perf = call_cerebras_api(
                prompt=prompt,
                api_key=cerebras_key,
                model=args.model,
                system_message=args.system_message,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                top_p=args.top_p,
                seed=args.seed,
                stream=args.stream,
                track_timing=True,
            )
            cerebras_content = extract_content(cerebras_response)
            
            # Generate with Anthropic
            print("[2/4] Generating with Anthropic...", file=sys.stderr)
            anthropic_response, anthropic_perf = call_anthropic_api(
                prompt=prompt,
                api_key=anthropic_key,
                model=args.anthropic_model,
                system_message=args.system_message,
                temperature=args.temperature,
                max_tokens=args.max_completion_tokens if args.max_completion_tokens > 0 else 4096,
                track_timing=True,
            )
            anthropic_content = extract_anthropic_content(anthropic_response)
            
            # Score both outputs using Anthropic
            print("[3/4] Scoring Cerebras output with Anthropic...", file=sys.stderr)
            cerebras_score_response, cerebras_score_perf = score_code(
                code=cerebras_content,
                prompt=prompt,
                api_key=anthropic_key,
                model=args.anthropic_model,
                language=args.language,
                track_timing=True,
                use_anthropic=True,
            )
            cerebras_scores = parse_score_response(cerebras_score_response, is_anthropic=True)
            
            print("[4/4] Scoring Anthropic output with Anthropic...", file=sys.stderr)
            anthropic_score_response, anthropic_score_perf = score_code(
                code=anthropic_content,
                prompt=prompt,
                api_key=anthropic_key,
                model=args.anthropic_model,
                language=args.language,
                track_timing=True,
                use_anthropic=True,
            )
            anthropic_scores = parse_score_response(anthropic_score_response, is_anthropic=True)
            
            # Display Performance Comparison Table
            print("\n" + "="*95, file=sys.stderr)
            print("⚡ CODE GENERATION PERFORMANCE COMPARISON", file=sys.stderr)
            print("="*95, file=sys.stderr)
            print(f"{'Metric':<35} {'Cerebras':<18} {'Anthropic':<18} {'Speed Up':<12} {'Winner':<10}", file=sys.stderr)
            print("-" * 93, file=sys.stderr)
            
            if cerebras_perf and anthropic_perf:
                perf_metrics = [
                    ("Time To First Token (TTFT)", "ttft_seconds", 1000, "ms", True),  # Lower is better
                    ("Latency Between Tokens", "token_latency_seconds", 1000, "ms/token", True),  # Lower is better
                    ("Input Throughput", "input_throughput_tokens_per_sec", 1, "tokens/sec", False),  # Higher is better
                    ("Output Throughput", "output_throughput_tokens_per_sec", 1, "tokens/sec", False),  # Higher is better
                    ("Total Time", "total_time_seconds", 1, "seconds", True),  # Lower is better
                    ("Input Tokens", "input_tokens", 1, "tokens", False),  # Info only
                    ("Output Tokens", "output_tokens", 1, "tokens", False),  # Info only
                    ("Total Tokens", "total_tokens", 1, "tokens", False),  # Info only
                ]
                
                for metric_name, key, multiplier, unit, lower_is_better in perf_metrics:
                    c_val = cerebras_perf[key] * multiplier
                    g_val = anthropic_perf[key] * multiplier
                    
                    # Calculate speedup (how many times faster is the winner)
                    if lower_is_better:
                        if c_val < g_val:
                            winner = "Cerebras"
                            if c_val > 0:
                                speedup = g_val / c_val
                                speedup_display = f"{speedup:.2f}x (C)"
                            else:
                                speedup_display = "N/A"
                        elif g_val < c_val:
                            winner = "Anthropic"
                            if g_val > 0:
                                speedup = c_val / g_val
                                speedup_display = f"{speedup:.2f}x (G)"
                            else:
                                speedup_display = "N/A"
                        else:
                            winner = "Tie"
                            speedup_display = "1.00x"
                    else:
                        if c_val > g_val:
                            winner = "Cerebras"
                            if g_val > 0:
                                speedup = c_val / g_val
                                speedup_display = f"{speedup:.2f}x (C)"
                            else:
                                speedup_display = "N/A"
                        elif g_val > c_val:
                            winner = "Anthropic"
                            if c_val > 0:
                                speedup = g_val / c_val
                                speedup_display = f"{speedup:.2f}x (G)"
                            else:
                                speedup_display = "N/A"
                        else:
                            winner = "Tie"
                            speedup_display = "1.00x"
                    
                    c_display = f"{c_val:.2f} {unit}"
                    g_display = f"{g_val:.2f} {unit}"
                    print(f"{metric_name:<35} {c_display:<18} {g_display:<18} {speedup_display:<12} {winner:<10}", file=sys.stderr)
            
            # Display Code Score Comparison Table
            print("\n" + "="*80, file=sys.stderr)
            print("📊 CODE QUALITY SCORE COMPARISON", file=sys.stderr)
            print("="*80, file=sys.stderr)
            print(f"{'Metric':<35} {'Cerebras':<20} {'Anthropic':<20} {'Winner':<10}", file=sys.stderr)
            print("-" * 85, file=sys.stderr)
            
            score_metrics = [
                ("Correctness (weight: 0.30)", "correctness", 0.30),
                ("Code Quality (weight: 0.30)", "code_quality", 0.30),
                ("Efficiency (weight: 0.20)", "efficiency", 0.20),
                ("Documentation (weight: 0.20)", "documentation", 0.20),
                ("TOTAL SCORE", "total_score", 1.0),
            ]
            
            for metric_name, key, weight in score_metrics:
                c_val = cerebras_scores[key]
                g_val = anthropic_scores[key]
                
                if weight < 1.0:
                    c_weighted = c_val * weight
                    g_weighted = g_val * weight
                    c_display = f"{c_val:.3f} (×{weight:.2f} = {c_weighted:.3f})"
                    g_display = f"{g_val:.3f} (×{weight:.2f} = {g_weighted:.3f})"
                else:
                    c_display = f"{c_val:.3f}"
                    g_display = f"{g_val:.3f}"
                
                winner = "Cerebras" if c_val > g_val else "Anthropic" if g_val > c_val else "Tie"
                print(f"{metric_name:<35} {c_display:<20} {g_display:<20} {winner:<10}", file=sys.stderr)
            
            # Display comprehensive summary
            print("\n" + "="*95, file=sys.stderr)
            print("📋 COMPREHENSIVE SUMMARY", file=sys.stderr)
            print("="*95, file=sys.stderr)
            
            # Performance summary
            if cerebras_perf and anthropic_perf:
                total_time_speedup = anthropic_perf['total_time_seconds'] / cerebras_perf['total_time_seconds'] if cerebras_perf['total_time_seconds'] > 0 else 0
                output_throughput_speedup = cerebras_perf['output_throughput_tokens_per_sec'] / anthropic_perf['output_throughput_tokens_per_sec'] if anthropic_perf['output_throughput_tokens_per_sec'] > 0 else 0
                
                print("\n⚡ PERFORMANCE SUMMARY:", file=sys.stderr)
                print(f"  • Cerebras generated {cerebras_perf['output_tokens']} tokens in {cerebras_perf['total_time_seconds']:.2f}s ({cerebras_perf['output_throughput_tokens_per_sec']:.2f} tok/s)", file=sys.stderr)
                print(f"  • Anthropic generated {anthropic_perf['output_tokens']} tokens in {anthropic_perf['total_time_seconds']:.2f}s ({anthropic_perf['output_throughput_tokens_per_sec']:.2f} tok/s)", file=sys.stderr)
                print(f"  • Cerebras is {total_time_speedup:.2f}x faster in total time", file=sys.stderr)
                print(f"  • Cerebras has {output_throughput_speedup:.2f}x higher output throughput", file=sys.stderr)
            
            # Quality summary
            print("\n📊 QUALITY SUMMARY:", file=sys.stderr)
            print(f"  • Cerebras Total Score: {cerebras_scores['total_score']:.3f}/1.000", file=sys.stderr)
            print(f"    - Correctness: {cerebras_scores['correctness']:.3f} (weight: 0.30)", file=sys.stderr)
            print(f"    - Code Quality: {cerebras_scores['code_quality']:.3f} (weight: 0.30)", file=sys.stderr)
            print(f"    - Efficiency: {cerebras_scores['efficiency']:.3f} (weight: 0.20)", file=sys.stderr)
            print(f"    - Documentation: {cerebras_scores['documentation']:.3f} (weight: 0.20)", file=sys.stderr)
            print(f"  • Anthropic Total Score: {anthropic_scores['total_score']:.3f}/1.000", file=sys.stderr)
            print(f"    - Correctness: {anthropic_scores['correctness']:.3f} (weight: 0.30)", file=sys.stderr)
            print(f"    - Code Quality: {anthropic_scores['code_quality']:.3f} (weight: 0.30)", file=sys.stderr)
            print(f"    - Efficiency: {anthropic_scores['efficiency']:.3f} (weight: 0.20)", file=sys.stderr)
            print(f"    - Documentation: {anthropic_scores['documentation']:.3f} (weight: 0.20)", file=sys.stderr)
            
            score_diff = anthropic_scores['total_score'] - cerebras_scores['total_score']
            if abs(score_diff) < 0.01:
                quality_winner = "Tie"
            elif score_diff > 0:
                quality_winner = f"Anthropic (+{score_diff:.3f})"
            else:
                quality_winner = f"Cerebras (+{abs(score_diff):.3f})"
            
            print(f"  • Quality Winner: {quality_winner}", file=sys.stderr)
            
            # Feedback summary
            print("\n💬 FEEDBACK:", file=sys.stderr)
            print(f"  • Cerebras: {cerebras_scores['feedback']}", file=sys.stderr)
            print(f"  • Anthropic: {anthropic_scores['feedback']}", file=sys.stderr)
            
            # Save comprehensive outputs to files
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure all outputs are saved under the output directory
            # Get absolute path to output directory to ensure consistency
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_base_dir = os.path.join(script_dir, "output")
            os.makedirs(output_base_dir, exist_ok=True)
            
            cerebras_file = os.path.join(output_base_dir, f"cerebras_output_{timestamp}.txt")
            anthropic_file = os.path.join(output_base_dir, f"anthropic_output_{timestamp}.txt")
            
            try:
                # Save Cerebras output with all details
                with open(cerebras_file, 'w', encoding='utf-8') as f:
                    f.write("="*95 + "\n")
                    f.write("CEREBRAS CODE GENERATION OUTPUT\n")
                    f.write("="*95 + "\n\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Model: {args.model}\n")
                    f.write(f"Language: {args.language}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("PERFORMANCE METRICS\n")
                    f.write("="*95 + "\n")
                    if cerebras_perf:
                        f.write(f"Time To First Token (TTFT):     {cerebras_perf['ttft_seconds']*1000:.2f} ms\n")
                        f.write(f"Latency Between Tokens:        {cerebras_perf['token_latency_seconds']*1000:.2f} ms/token\n")
                        f.write(f"Input Throughput:               {cerebras_perf['input_throughput_tokens_per_sec']:.2f} tokens/sec\n")
                        f.write(f"Output Throughput:              {cerebras_perf['output_throughput_tokens_per_sec']:.2f} tokens/sec\n")
                        f.write(f"Total Time:                     {cerebras_perf['total_time_seconds']:.3f} seconds\n")
                        f.write(f"Input Tokens:                   {cerebras_perf['input_tokens']}\n")
                        f.write(f"Output Tokens:                  {cerebras_perf['output_tokens']}\n")
                        f.write(f"Total Tokens:                   {cerebras_perf['total_tokens']}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("CODE QUALITY SCORES\n")
                    f.write("="*95 + "\n")
                    f.write(f"Correctness (0.30):  {cerebras_scores['correctness']:.3f} × 0.30 = {cerebras_scores['correctness'] * 0.30:.3f}\n")
                    f.write(f"Code Quality (0.30): {cerebras_scores['code_quality']:.3f} × 0.30 = {cerebras_scores['code_quality'] * 0.30:.3f}\n")
                    f.write(f"Efficiency (0.20):   {cerebras_scores['efficiency']:.3f} × 0.20 = {cerebras_scores['efficiency'] * 0.20:.3f}\n")
                    f.write(f"Documentation (0.20): {cerebras_scores['documentation']:.3f} × 0.20 = {cerebras_scores['documentation'] * 0.20:.3f}\n")
                    f.write(f"\nTOTAL SCORE: {cerebras_scores['total_score']:.3f} / 1.000\n")
                    f.write(f"\nFeedback: {cerebras_scores['feedback']}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("GENERATED CODE\n")
                    f.write("="*95 + "\n\n")
                    f.write(cerebras_content)
                
                # Save Anthropic output with all details
                with open(anthropic_file, 'w', encoding='utf-8') as f:
                    f.write("="*95 + "\n")
                    f.write("ANTHROPIC CODE GENERATION OUTPUT\n")
                    f.write("="*95 + "\n\n")
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Model: {args.anthropic_model}\n")
                    f.write(f"Language: {args.language}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("PERFORMANCE METRICS\n")
                    f.write("="*95 + "\n")
                    if anthropic_perf:
                        f.write(f"Time To First Token (TTFT):     {anthropic_perf['ttft_seconds']*1000:.2f} ms\n")
                        f.write(f"Latency Between Tokens:        {anthropic_perf['token_latency_seconds']*1000:.2f} ms/token\n")
                        f.write(f"Input Throughput:               {anthropic_perf['input_throughput_tokens_per_sec']:.2f} tokens/sec\n")
                        f.write(f"Output Throughput:              {anthropic_perf['output_throughput_tokens_per_sec']:.2f} tokens/sec\n")
                        f.write(f"Total Time:                     {anthropic_perf['total_time_seconds']:.3f} seconds\n")
                        f.write(f"Input Tokens:                   {anthropic_perf['input_tokens']}\n")
                        f.write(f"Output Tokens:                  {anthropic_perf['output_tokens']}\n")
                        f.write(f"Total Tokens:                   {anthropic_perf['total_tokens']}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("CODE QUALITY SCORES\n")
                    f.write("="*95 + "\n")
                    f.write(f"Correctness (0.30):  {anthropic_scores['correctness']:.3f} × 0.30 = {anthropic_scores['correctness'] * 0.30:.3f}\n")
                    f.write(f"Code Quality (0.30): {anthropic_scores['code_quality']:.3f} × 0.30 = {anthropic_scores['code_quality'] * 0.30:.3f}\n")
                    f.write(f"Efficiency (0.20):   {anthropic_scores['efficiency']:.3f} × 0.20 = {anthropic_scores['efficiency'] * 0.20:.3f}\n")
                    f.write(f"Documentation (0.20): {anthropic_scores['documentation']:.3f} × 0.20 = {anthropic_scores['documentation'] * 0.20:.3f}\n")
                    f.write(f"\nTOTAL SCORE: {anthropic_scores['total_score']:.3f} / 1.000\n")
                    f.write(f"\nFeedback: {anthropic_scores['feedback']}\n")
                    f.write("\n" + "="*95 + "\n")
                    f.write("GENERATED CODE\n")
                    f.write("="*95 + "\n\n")
                    f.write(anthropic_content)
                
                print(f"\n💾 OUTPUT FILES SAVED:", file=sys.stderr)
                print(f"  • Cerebras: {cerebras_file}", file=sys.stderr)
                print(f"  • Anthropic: {anthropic_file}", file=sys.stderr)
                print(f"\n📄 All details (performance, scores, code) have been saved to the files above.", file=sys.stderr)
            except Exception as e:
                print(f"\n⚠️  Warning: Could not save output files: {e}", file=sys.stderr)
            
            if args.json:
                output_data = {
                    "cerebras": {
                        "content": cerebras_content,
                        "scores": cerebras_scores,
                        "performance": cerebras_perf,
                    },
                    "anthropic": {
                        "content": anthropic_content,
                        "scores": anthropic_scores,
                        "performance": anthropic_perf,
                    },
                }
                print("\n" + json.dumps(output_data, indent=2))
            
        except Exception as e:
            print(f"Error in comparison: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:
        # Regular single API mode
        # Get API key
        api_key = args.api_key or os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            print("Error: CEREBRAS_API_KEY environment variable not set and --api-key not provided", file=sys.stderr)
            sys.exit(1)
        
        # Call API
        try:
            response, perf_metrics = call_cerebras_api(
                prompt=prompt,
                api_key=api_key,
                model=args.model,
                system_message=args.system_message,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
                top_p=args.top_p,
                seed=args.seed,
                stream=args.stream,
                track_timing=True,  # Always track timing for generation
            )
            
            if args.json:
                output_data = {"response": response}
                if perf_metrics:
                    output_data["performance"] = perf_metrics
                print(json.dumps(output_data, indent=2))
            else:
                content = extract_content(response)
                print(content)
            
            # Display performance metrics
            if perf_metrics:
                print("\n" + "="*60, file=sys.stderr)
                print("⚡ PERFORMANCE METRICS", file=sys.stderr)
                print("="*60, file=sys.stderr)
                print(f"  Time To First Token (TTFT):     {perf_metrics['ttft_seconds']*1000:.2f} ms", file=sys.stderr)
                print(f"  Latency Between Tokens:        {perf_metrics['token_latency_seconds']*1000:.2f} ms/token", file=sys.stderr)
                print(f"  Input Throughput:               {perf_metrics['input_throughput_tokens_per_sec']:.2f} tokens/sec", file=sys.stderr)
                print(f"  Output Throughput:              {perf_metrics['output_throughput_tokens_per_sec']:.2f} tokens/sec", file=sys.stderr)
                print(f"\n  Total Time:                     {perf_metrics['total_time_seconds']:.3f} seconds", file=sys.stderr)
                print(f"  Input Tokens:                   {perf_metrics['input_tokens']}", file=sys.stderr)
                print(f"  Output Tokens:                  {perf_metrics['output_tokens']}", file=sys.stderr)
                print(f"  Total Tokens:                   {perf_metrics['total_tokens']}", file=sys.stderr)
            
            # Score the code if requested
            if args.score:
                print("\n" + "="*60, file=sys.stderr)
                print("SCORING CODE...", file=sys.stderr)
                print("="*60, file=sys.stderr)
                
                try:
                    # Use Anthropic for scoring
                    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                    if not anthropic_key:
                        print("Warning: ANTHROPIC_API_KEY not set, using Cerebras for scoring", file=sys.stderr)
                        score_model = args.score_model or args.model
                        use_anthropic = False
                        score_api_key = api_key
                    else:
                        score_model = args.anthropic_model
                        use_anthropic = True
                        score_api_key = anthropic_key
                    
                    score_response, score_perf_metrics = score_code(
                        code=content,
                        prompt=prompt,
                        api_key=score_api_key,
                        model=score_model,
                        language=args.language,
                        track_timing=True,
                        use_anthropic=use_anthropic,
                    )
                    
                    scores = parse_score_response(score_response, is_anthropic=use_anthropic)
                    
                    # Display scores
                    print("\n📊 CODE SCORES:", file=sys.stderr)
                    print(f"  Correctness (0.30):  {scores['correctness']:.3f} × 0.30 = {scores['correctness'] * 0.30:.3f}", file=sys.stderr)
                    print(f"  Code Quality (0.30): {scores['code_quality']:.3f} × 0.30 = {scores['code_quality'] * 0.30:.3f}", file=sys.stderr)
                    print(f"  Efficiency (0.20):  {scores['efficiency']:.3f} × 0.20 = {scores['efficiency'] * 0.20:.3f}", file=sys.stderr)
                    print(f"  Documentation (0.20): {scores['documentation']:.3f} × 0.20 = {scores['documentation'] * 0.20:.3f}", file=sys.stderr)
                    print(f"\n  TOTAL SCORE: {scores['total_score']:.3f} / 1.000", file=sys.stderr)
                    print(f"\n💬 Feedback: {scores['feedback']}", file=sys.stderr)
                    
                    # Display scoring performance metrics
                    if score_perf_metrics:
                        print("\n" + "="*60, file=sys.stderr)
                        print("⚡ SCORING PERFORMANCE METRICS", file=sys.stderr)
                        print("="*60, file=sys.stderr)
                        print(f"  Time To First Token (TTFT):     {score_perf_metrics['ttft_seconds']*1000:.2f} ms", file=sys.stderr)
                        print(f"  Latency Between Tokens:        {score_perf_metrics['token_latency_seconds']*1000:.2f} ms/token", file=sys.stderr)
                        print(f"  Input Throughput:               {score_perf_metrics['input_throughput_tokens_per_sec']:.2f} tokens/sec", file=sys.stderr)
                        print(f"  Output Throughput:              {score_perf_metrics['output_throughput_tokens_per_sec']:.2f} tokens/sec", file=sys.stderr)
                        print(f"\n  Total Time:                     {score_perf_metrics['total_time_seconds']:.3f} seconds", file=sys.stderr)
                        print(f"  Input Tokens:                   {score_perf_metrics['input_tokens']}", file=sys.stderr)
                        print(f"  Output Tokens:                  {score_perf_metrics['output_tokens']}", file=sys.stderr)
                        print(f"  Total Tokens:                   {score_perf_metrics['total_tokens']}", file=sys.stderr)
                    
                    if args.json:
                        output_data = {"scores": scores}
                        if score_perf_metrics:
                            output_data["scoring_performance"] = score_perf_metrics
                        print("\n" + json.dumps(output_data, indent=2))
                        
                except Exception as e:
                    print(f"Error scoring code: {e}", file=sys.stderr)
                    # Don't exit on scoring errors, just warn
                    import traceback
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
