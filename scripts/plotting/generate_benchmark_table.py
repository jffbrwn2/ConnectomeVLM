#!/usr/bin/env python3
"""
Generate LaTeX table in ICML style summarizing proofreading benchmark results.

This script extracts balanced accuracy scores from evaluation results and generates
a formatted LaTeX table suitable for inclusion in an ICML-style paper.

Usage:
    cd reproduction/
    python scripts/plotting/generate_benchmark_table.py
"""

import json
import os
from pathlib import Path
import numpy as np

# Default path: relative to reproduction/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # reproduction/
DATA_DIR = REPO_DIR / "data"


def compute_balanced_accuracy_from_predictions(predictions):
    """Calculate balanced accuracy from prediction list."""
    tp = sum(1 for p in predictions if p['ground_truth'] and p['predicted'])
    tn = sum(1 for p in predictions if not p['ground_truth'] and not p['predicted'])
    fp = sum(1 for p in predictions if not p['ground_truth'] and p['predicted'])
    fn = sum(1 for p in predictions if p['ground_truth'] and not p['predicted'])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (tpr + tnr) / 2

    return balanced_acc


def bootstrap_balanced_accuracy(predictions, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval for balanced accuracy."""
    np.random.seed(42)

    n = len(predictions)
    bootstrap_accs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample = [predictions[i] for i in indices]

        # Compute balanced accuracy for this sample
        acc = compute_balanced_accuracy_from_predictions(sample)
        bootstrap_accs.append(acc)

    # Compute standard deviation
    std = np.std(bootstrap_accs)

    return std


def compute_balanced_accuracy_from_human(responses):
    """Calculate balanced accuracy and standard error from human evaluation responses."""
    tp = sum(1 for r in responses if r['ground_truth'] and r['human_answer'])
    tn = sum(1 for r in responses if not r['ground_truth'] and not r['human_answer'])
    fp = sum(1 for r in responses if not r['ground_truth'] and r['human_answer'])
    fn = sum(1 for r in responses if r['ground_truth'] and not r['human_answer'])

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (tpr + tnr) / 2

    # Compute standard error
    n_pos = tp + fn
    n_neg = tn + fp

    # Standard error for TPR and TNR
    se_tpr = np.sqrt(tpr * (1 - tpr) / n_pos) if n_pos > 0 else 0
    se_tnr = np.sqrt(tnr * (1 - tnr) / n_neg) if n_neg > 0 else 0

    # Standard error for balanced accuracy (average of TPR and TNR)
    se_balanced = np.sqrt((se_tpr**2 + se_tnr**2) / 4)

    return balanced_acc, se_balanced


def load_results():
    """Load all evaluation results and compute balanced accuracies."""

    base_path = DATA_DIR

    results = {
        'merge_action': {},
        'endpoint_error_identification_with_em': {},
        'merge_error_identification': {},
        'split_action': {}
    }

    # Human baseline
    human_files = {
        'merge_action': base_path / "human_eval/human_eval_split-error-correction_anonymous_453d4b109747.json",
        'endpoint_error_identification_with_em': base_path / "human_eval/human_eval_endpoint_error_identification_with_em_anonymous_12149c9ecfa7.json",
        'merge_error_identification': base_path / "human_eval/human_eval_merge-error-identification_anonymous_29e820ea8eb1.json",
        'split_action': base_path / "human_eval/human_eval_split-correction-evaluation_anonymous_ef37eba8df98.json"
    }

    for task, filepath in human_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        acc, err = compute_balanced_accuracy_from_human(data['responses'])
        results[task]['Human'] = {'acc': acc, 'err': err}

    # Linear Probe (SigLIP-2)
    linear_probe_files = {
        'merge_action': base_path / "final_data/linear_probe_google_siglip2-so400m-patch16-512_merge_action_merge-parquet_512samples.json",
        'merge_error_identification': base_path / "final_data/linear_probe_google_siglip2-so400m-patch16-512_merge_error_identification_merge-error-identification-parquet_512samples.json",
        'split_action': base_path / "final_data/linear_probe_google_siglip2-so400m-patch16-512_split_action_splits-parquet_512samples.json",
        'endpoint_error_identification_with_em': base_path / "final_data/linear_probe_google_siglip2-so400m-patch16-512_endpoint_error_identification_with_em_endpoints-with-em-parquet_4626samples.json"
    }

    for task, filepath in linear_probe_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        val_tpr = data.get('val_tpr', 0)
        val_tnr = data.get('val_tnr', 0)
        acc = (val_tpr + val_tnr) / 2

        # Use CV std for error
        cv_data = data.get('cross_validation', {})
        cv_tpr_std = cv_data.get('cv_tpr_std', 0)
        cv_tnr_std = cv_data.get('cv_tnr_std', 0)
        # Error propagation: std(balanced_acc) = sqrt((std_tpr^2 + std_tnr^2) / 4)
        err = np.sqrt((cv_tpr_std**2 + cv_tnr_std**2) / 4)

        results[task]['Linear Probe'] = {'acc': acc, 'err': err}

    # ResNet-50
    resnet_file = base_path / "final_data/resnet_results.json"
    with open(resnet_file, 'r') as f:
        resnet_data = json.load(f)

    resnet_task_map = {
        'merge_action': 'merge_action_resnet',
        'merge_error_identification': 'merge_error_identification_resnet',
        'split_action': 'split_action_resnet',
        'endpoint_error_identification_with_em': 'endpoint_error_identification_with_em_resnet'
    }

    # For ResNet, we need to load the actual evaluation files to get predictions for bootstrapping
    resnet_eval_files = {
        'merge_action': base_path / "final_data/eval_merge_action__checkpoints_merge_action_finetune_Qwen3-VL-32B-Instruct_merge_action_lora_merger_20260122_213833_samplesall_epochs3_lr0.0002_r16_merged_test_votes5_20260125_195801.json",
        'merge_error_identification': base_path / "final_data/eval_merge_error_identification__checkpoints_merge_error_identification_finetune_Qwen3-VL-32B-Instruct_merge_error_identification_lora_merger_20260122_145540_samplesall_epochs3_lr0.0002_r16_merged_test_votes5_20260122_163316.json",
        'split_action': base_path / "final_data/eval_split_action_Qwen3-VL-32B-Instruct_split_action_32B_r32_longer_patience_checkpoint-900_merged_test_votes25_20260119_020602.json",
        'endpoint_error_identification_with_em': base_path / "final_data/eval_endpoint_error_identification_with_em__checkpoints_endpoint_error_identification_with_em_finetune_Qwen3-VL-32B-Instruct_endpoint_identification_EM_lora_merger_big_20260126_012931_samplesall_epochs3_lr8e-05_r16_merged_test_votes25_20260126_132940.json"
    }

    for task, resnet_key in resnet_task_map.items():
        if resnet_key in resnet_data:
            # Get balanced accuracy from summary
            finetuned_acc = resnet_data[resnet_key].get('fully_finetuned_model', {}).get('balanced_accuracy', 0)

            # Load predictions for bootstrap error
            if task in resnet_eval_files and resnet_eval_files[task].exists():
                with open(resnet_eval_files[task], 'r') as f:
                    eval_data = json.load(f)
                if 'predictions' in eval_data:
                    finetuned_err = bootstrap_balanced_accuracy(eval_data['predictions'])
                else:
                    finetuned_err = 0.0
            else:
                finetuned_err = 0.0

            results[task]['ResNet-50 Finetuned'] = {'acc': finetuned_acc, 'err': finetuned_err}

    # VLM (Qwen3-VL-32B) - Use derived accuracy when available (from vlm_evaluations_with_derived),
    # otherwise use majority voting results with 25 votes from vlm_evaluations/
    vlm_files = {
        'merge_action': {
            'path': base_path / "final_data/vlm_evaluations_with_derived/eval_merge_action_merge_action_finetune_Qwen3-VL-32B-Instruct_merge_action_lora_merger_20260126_002129_samplesall_epochs3_lr0.0002_r16_merged_test_votes5_20260128_020400.json",
            'prefer_derived': True
        },
        'merge_error_identification': {
            'path': base_path / "final_data/vlm_evaluations/eval_merge_error_identification_merge_error_identification_finetune_Qwen3-VL-32B-Instruct_merge_error_identification_lora_merger_20260122_145540_samplesall_epochs3_lr0.0002_r16_merged_test_votes25_20260128_040051.json",
            'prefer_derived': False
        },
        'split_action': {
            'path': base_path / "final_data/vlm_evaluations/eval_split_action_Qwen3-VL-32B-Instruct_split_action_32B_r32_longer_patience_checkpoint-900_merged_test_votes25_20260128_214153.json",
            'prefer_derived': False
        },
        'endpoint_error_identification_with_em': {
            'path': base_path / "final_data/vlm_evaluations/eval_endpoint_error_identification_with_em_endpoint_error_identification_with_em_finetune_Qwen3-VL-32B-Instruct_endpoint_identification_EM_lora_merger_big_20260126_012931_samplesall_epochs3_lr8e-05_r16_merged_test_votes25_20260128_222232.json",
            'prefer_derived': False
        }
    }

    for task, file_info in vlm_files.items():
        with open(file_info['path'], 'r') as f:
            data = json.load(f)

        # Prefer derived_accuracy if available and requested
        if file_info['prefer_derived'] and 'derived_accuracy' in data:
            acc = data['derived_accuracy']
        elif 'majority_accuracy' in data:
            acc = data['majority_accuracy']
        elif 'balanced_accuracy' in data:
            acc = data['balanced_accuracy']
        else:
            acc = compute_balanced_accuracy_from_predictions(data['predictions'])

        # Bootstrap error from predictions
        if 'predictions' in data:
            err = bootstrap_balanced_accuracy(data['predictions'])
        else:
            err = 0.0

        results[task]['VLM'] = {'acc': acc, 'err': err}

    # GPT-5
    gpt5_files = {
        'merge_action': base_path / "final_data/eval_merge_action_gpt-5_votes5_20260122_131714.json",
        'merge_error_identification': base_path / "final_data/eval_merge_error_identification_gpt-5_votes5_20260124_115258.json",
        'split_action': base_path / "final_data/eval_split_action_gpt-5_votes5_20260124_115305.json",
        'endpoint_error_identification_with_em': base_path / "final_data/eval_endpoint_error_identification_with_em_gpt-5_votes5_20260127_181340.json"
    }

    for task, filepath in gpt5_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        acc = compute_balanced_accuracy_from_predictions(data['predictions'])
        err = bootstrap_balanced_accuracy(data['predictions'])
        results[task]['GPT-5'] = {'acc': acc, 'err': err}

    # Gemini-3-Pro-Preview
    gemini_files = {
        'merge_action': base_path / "final_data/eval_merge_action_google_gemini-3-pro-preview_votes5_20260122_151652.json",
        'merge_error_identification': base_path / "final_data/eval_merge_error_identification_google_gemini-3-pro-preview_votes5_20260124_125911.json",
        'split_action': base_path / "final_data/eval_split_action_google_gemini-3-pro-preview_votes5_20260124_115751.json",
        'endpoint_error_identification_with_em': base_path / "final_data/eval_endpoint_error_identification_with_em_google_gemini-3-pro-preview_votes5_20260127_173914.json"
    }

    for task, filepath in gemini_files.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
        acc = compute_balanced_accuracy_from_predictions(data['predictions'])
        err = bootstrap_balanced_accuracy(data['predictions'])
        results[task]['Gemini-3-Pro'] = {'acc': acc, 'err': err}

    return results


def generate_latex_table(results):
    """Generate LaTeX table in ICML style."""

    # Task display names (shortened for table)
    task_names = {
        'merge_action': 'Split Error Correction',
        'endpoint_error_identification_with_em': 'Split Error Identification',
        'merge_error_identification': 'Merge Error Identification',
        'split_action': 'Split Action Evaluation'
    }

    # Model column order (removed ResNet-50 Frozen)
    models = [
        'Human',
        'Linear Probe',
        'ResNet-50 Finetuned',
        'VLM',
        'GPT-5',
        'Gemini-3-Pro'
    ]

    # Find best model for each task (for bolding)
    best_models = {}
    for task in results.keys():
        max_acc = 0
        best_model = None
        for model in models:
            if model in results[task]:
                acc = results[task][model]['acc']
                if acc > max_acc:
                    max_acc = acc
                    best_model = model
        best_models[task] = best_model

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\caption{Proofreading benchmark results showing balanced accuracy (\\%) across four tasks. "
                 "Bold indicates best model per task. Human baseline represents expert annotator performance. "
                 "All results evaluated on the MICrONS mouse cortex dataset.}")
    latex.append("\\label{tab:benchmark}")
    latex.append("\\vskip 0.15in")
    latex.append("\\begin{center}")
    latex.append("\\begin{small}")
    latex.append("\\begin{sc}")

    # Table header
    header = "\\begin{tabular}{l" + "c" * len(models) + "}"
    latex.append(header)
    latex.append("\\toprule")

    # Column headers
    model_headers = " & ".join(["Task"] + models) + " \\\\"
    latex.append(model_headers)
    latex.append("\\midrule")

    # Data rows
    for task in ['merge_action', 'endpoint_error_identification_with_em', 'merge_error_identification', 'split_action']:
        row_data = [task_names[task]]
        for model in models:
            if model in results[task]:
                acc = results[task][model]['acc'] * 100  # Convert to percentage
                err = results[task][model]['err'] * 100  # Convert to percentage
                acc_str = f"{acc:.1f}$\\pm${err:.1f}"
                # Bold the best model
                if model == best_models[task]:
                    acc_str = f"\\textbf{{{acc_str}}}"
                row_data.append(acc_str)
            else:
                row_data.append("--")

        latex.append(" & ".join(row_data) + " \\\\")

    # Table footer
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{sc}")
    latex.append("\\end{small}")
    latex.append("\\end{center}")
    latex.append("\\vskip -0.1in")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main():
    """Main function to generate and save the LaTeX table."""
    print("Loading evaluation results...")
    results = load_results()

    print("\nSummary of loaded results:")
    for task, task_results in results.items():
        print(f"\n{task}:")
        for model, data in task_results.items():
            acc = data['acc'] * 100
            err = data['err'] * 100
            print(f"  {model}: {acc:.1f}% +/- {err:.1f}%")

    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(results)

    # Save to file
    output_path = REPO_DIR / "output" / "benchmark_table.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"\nLaTeX table saved to: {output_path}")
    print("\nGenerated LaTeX:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    return latex_table


if __name__ == "__main__":
    main()
