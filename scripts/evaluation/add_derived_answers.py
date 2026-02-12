#!/usr/bin/env python3
"""
Add derived answers to evaluation JSON files.

This script processes VLM evaluation files, extracts the analysis from each response,
uses an LLM to derive what answer the analysis supports, and adds this as a new key
in the predictions.

Usage:
    python scripts/analysis/add_derived_answers.py \
        --input evaluation_results/final_data/vlm_evaluations/*.json \
        --model gpt-4o-mini \
        --output-dir evaluation_results/final_data/vlm_evaluations_with_derived
"""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Optional, List
import sys

# Add parent directory to path for util import
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_analysis(response: str) -> Optional[str]:
    """Extract the analysis text from a model response."""
    if not response:
        return None

    # Try to find analysis tags
    match = re.search(r'<analysis>(.*?)</analysis>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # If no tags, try to extract everything before the answer tag
    answer_match = re.search(r'<answer>', response, re.IGNORECASE)
    if answer_match:
        return response[:answer_match.start()].strip()

    return None


def extract_answer(response: str) -> Optional[bool]:
    """Extract yes/no answer from response."""
    if not response:
        return None

    match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response.lower())
    if match:
        return match.group(1) == "yes"
    return None


def create_consistency_prompt(analysis: str, task_name: str) -> str:
    """Create a prompt to derive an answer from just the analysis."""
    # Determine if this is a merge or split task
    is_merge = "merge" in task_name.lower()

    return f"""You are evaluating whether two neural segments should be {'merged' if is_merge else 'split'} based on an analysis.

The following analysis was written by an expert examining 3D views of {'two segments (blue = original, orange = candidate merge)' if is_merge else 'a single split into blue and green segments'}:

<analysis>
{analysis}
</analysis>

Based ONLY on this analysis text, what answer does the reasoning support?

- yes = The analysis supports {'merging' if is_merge else 'splitting'} (segments should be connected)
- no = The analysis supports {'keeping them separate' if is_merge else 'keeping them together'}

Important: Base your answer strictly on what the analysis says, not on what you think the correct answer might be.

Surround your answer with <answer> and </answer> tags."""


async def add_derived_answers_to_file(
    input_file: Path,
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    """
    Process an evaluation file and add derived answers.

    Args:
        input_file: Path to evaluation JSON file
        model: LLM model to use for deriving answers
        batch_size: Batch size for API calls
        output_dir: Output directory (if None, overwrites input)
        overwrite: If True, overwrite input file; if False and output_dir is None, error
    """
    from util import LLMProcessor

    print(f"Processing: {input_file}")

    # Load evaluation results
    with open(input_file, 'r') as f:
        eval_results = json.load(f)

    task_name = eval_results.get('task_name', '')
    predictions = eval_results.get('predictions', [])

    print(f"  Task: {task_name}")
    print(f"  Predictions: {len(predictions)}")

    # Collect all responses that need analysis
    analysis_jobs = []  # List of (pred_idx, vote_idx, analysis, original_answer)

    for pred_idx, pred in enumerate(predictions):
        all_responses = pred.get('all_responses', [pred.get('response', '')])
        all_predictions = pred.get('all_predictions', [pred.get('predicted')])

        for vote_idx, (response, original_answer) in enumerate(zip(all_responses, all_predictions)):
            analysis = extract_analysis(response)
            if analysis:
                analysis_jobs.append({
                    'pred_idx': pred_idx,
                    'vote_idx': vote_idx,
                    'analysis': analysis,
                    'original_answer': original_answer,
                })

    print(f"  Responses with analysis: {len(analysis_jobs)}")

    if not analysis_jobs:
        print("  No analysis found in responses, skipping file")
        return

    # Initialize LLM
    llm = LLMProcessor(model=model, max_tokens=256, max_concurrent=batch_size)

    # Prepare prompts
    prompts = []
    for job in analysis_jobs:
        prompt = create_consistency_prompt(job['analysis'], task_name)
        prompts.append([{"role": "user", "content": prompt}])

    # Run inference
    print(f"  Running derived answer extraction with {model}...")
    responses = await llm.process_batch_conversations(prompts)

    # Parse derived answers
    derived_answers = []
    for response in responses:
        derived_answer = extract_answer(response)
        derived_answers.append(derived_answer)

    # Add derived answers back to predictions
    for job, derived_answer in zip(analysis_jobs, derived_answers):
        pred = predictions[job['pred_idx']]

        # Initialize derived answer storage if not present
        if 'all_derived_answers' not in pred:
            num_votes = len(pred.get('all_responses', [pred.get('response', '')]))
            pred['all_derived_answers'] = [None] * num_votes

        pred['all_derived_answers'][job['vote_idx']] = derived_answer

    # Compute derived majority vote for each prediction
    for pred in predictions:
        if 'all_derived_answers' in pred:
            derived_answers = [a for a in pred['all_derived_answers'] if a is not None]
            if derived_answers:
                # Majority vote
                from collections import Counter
                vote_counts = Counter(derived_answers)
                derived_majority = vote_counts.most_common(1)[0][0]
                pred['derived_predicted'] = derived_majority
                pred['derived_correct'] = (derived_majority == pred['ground_truth'])
            else:
                pred['derived_predicted'] = None
                pred['derived_correct'] = None

    # Compute overall derived accuracy
    derived_correct = sum(1 for p in predictions if p.get('derived_correct') is True)
    total_with_derived = sum(1 for p in predictions if 'derived_predicted' in p and p['derived_predicted'] is not None)

    if total_with_derived > 0:
        derived_accuracy = derived_correct / total_with_derived
        eval_results['derived_accuracy'] = derived_accuracy
        eval_results['derived_num_correct'] = derived_correct
        eval_results['derived_num_samples'] = total_with_derived

        print(f"  Original accuracy: {eval_results.get('accuracy', 0):.3f}")
        print(f"  Derived accuracy: {derived_accuracy:.3f} ({derived_correct}/{total_with_derived})")

    # Determine output path
    if output_dir:
        output_path = output_dir / input_file.name
        output_dir.mkdir(parents=True, exist_ok=True)
    elif overwrite:
        output_path = input_file
    else:
        raise ValueError("Must specify output_dir or set overwrite=True")

    # Save modified results
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)

    print(f"  Saved to: {output_path}")


async def process_multiple_files(
    input_files: List[Path],
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    """Process multiple evaluation files."""
    print(f"Processing {len(input_files)} files...\n")

    for input_file in input_files:
        try:
            await add_derived_answers_to_file(
                input_file,
                model=model,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            print()
        except Exception as e:
            print(f"  Error processing {input_file}: {e}")
            print()
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Add derived answers to evaluation JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        nargs='+',
        required=True,
        help="Path(s) to evaluation results JSON file(s) (supports wildcards)"
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="Model to use for deriving answers from analysis (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=20,
        help="Batch size for API calls (default: 20)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for modified files (default: overwrite input files)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files (only if --output-dir not specified)"
    )

    args = parser.parse_args()

    # Expand input files (handle wildcards)
    input_files = []
    for pattern in args.input:
        matching_files = list(Path().glob(pattern))
        if matching_files:
            input_files.extend(matching_files)
        else:
            # Try as literal path
            p = Path(pattern)
            if p.exists():
                input_files.append(p)
            else:
                print(f"Warning: No files found matching: {pattern}")

    if not input_files:
        print("Error: No input files found")
        return

    # Check output settings
    if not args.output_dir and not args.overwrite:
        print("Error: Must specify --output-dir or --overwrite")
        return

    asyncio.run(process_multiple_files(
        input_files=input_files,
        model=args.model,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
