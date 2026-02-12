# Reproduction Package

This directory contains the complete pipeline to reproduce the paper's results, from model training through evaluation to figure generation.

- **Quick path**: Generate figures from pre-computed data (`pip install matplotlib numpy && bash generate_all_figures.sh`)
- **Full path**: Re-run training and evaluation using the included scripts (requires Modal cloud GPU platform)

## Quick Start: Generate Figures

```bash
cd reproduction/
pip install -r requirements.txt
bash generate_all_figures.sh
```

Output files appear in `output/`.

## Figures Produced

| Output File | Paper Location | Description |
|---|---|---|
| `figure2_scaling_curves.png` | Figure 2 | Balanced accuracy vs training samples (2x2 grid, 4 tasks). Compares Linear Probe scaling curve with ResNet-50 (frozen, finetuned, small) and human baselines. |
| `figure3a_cross_species.png` | Figure 3A | Cross-species transfer (1x4 grouped bars). Shows balanced accuracy for ResNet, Linear Probe, and VLM across mouse, fly, zebrafish, and human datasets. |
| `figure3b_ece_vs_data.png` | Figure 3B | Expected Calibration Error (ECE) vs training samples (2x2 grid). Shows how calibration improves with more training data. |
| `figure3c_calibration.png` | Figure 3C | Cross-species calibration ECE (1x4 grouped bars). Shows ECE for each model across species. |
| `benchmark_table.tex` | Table 1 | LaTeX benchmark table with balanced accuracy for Human, Linear Probe, ResNet-50, VLM, GPT-5, and Gemini-3-Pro across all four tasks. |

All figure scripts also produce PDF versions alongside the PNG.

---

## Full Pipeline: From Training to Figures

The complete pipeline has four stages. Each stage's scripts are included in this package.

### Stage 1: Model Training (GPU required)

Training scripts are in `scripts/training/`. They run on [Modal](https://modal.com/) (cloud GPU platform) and require access to the training data volumes.

| Script | Purpose | GPU | Key Output |
|---|---|---|---|
| `modal_resnet_finetune.py` | Train ResNet-50 (frozen, finetuned, small) for all 4 tasks | A10G | Model checkpoints on Modal volume |
| `modal_qwen_finetune.py` | Fine-tune Qwen3-VL-32B with LoRA for all 4 tasks; also runs linear probe data sweeps | A100/H100 | LoRA adapters; linear probe sweep JSONs |
| `run_scaling_experiments.sh` | Orchestrate parallel training across tasks/models | - | Coordinates up to 5 parallel Modal jobs |

**Training commands** (examples):

```bash
# ResNet-50 finetuning (all tasks)
modal run scripts/training/modal_resnet_finetune.py::train \
    --task merge_action --model resnet50 --freeze-backbone false

# ResNet-50 frozen (feature extractor only)
modal run scripts/training/modal_resnet_finetune.py::train \
    --task merge_action --model resnet50 --freeze-backbone true

# VLM finetuning (Qwen3-VL-32B with LoRA)
modal run scripts/training/modal_qwen_finetune.py::train \
    --task merge_action --model Qwen3-VL-32B-Instruct \
    --epochs 3 --lr 0.0002 --lora-r 16

# Linear probe data sweep (produces scaling curve data)
modal run scripts/training/modal_qwen_finetune.py::data_sweep \
    --task merge_action --n-repeats 10 --seed 42
```

**Task definitions** are in `src/environment/task_configs.py` (4,050 lines). This defines all four benchmark tasks:
- `merge_action` (Split Error Correction)
- `endpoint_error_identification_with_em` (Split Error Identification)
- `merge_error_identification` (Merge Error Identification)
- `split_action` (Split Action Evaluation)

**Training data**: `src/training/question_dataset.py` defines the PyTorch Dataset used for ResNet and VLM training. It loads task-specific parquet files containing image grids and binary labels.

### Stage 2: Model Evaluation (GPU required)

Evaluation scripts are in `scripts/evaluation/`. They evaluate trained models on test sets and cross-species datasets.

| Script | Purpose | GPU | Key Output |
|---|---|---|---|
| `modal_proofreading_inference.py` | VLM inference backend (vLLM serving) | A100/H100 | Per-sample predictions with confidence |
| `proofreading_evaluator.py` | End-to-end evaluation pipeline for proofreading tasks | - | Evaluation JSONs with accuracy, predictions |
| `modal_linear_probe_cross_species.py` | Evaluate mouse-trained linear probes on fly/zebrafish/human data | A10G | Cross-species accuracy, ECE, MCE |
| `evaluate_cross_species.py` | Cross-species ResNet evaluation | - | Per-species accuracy metrics |
| `compute_resnet_calibration.py` | Compute ECE, MCE, Brier score for ResNet checkpoints | Optional | Calibration summary JSONs |
| `human_eval.py` | Gradio GUI for collecting human baselines | - | Human evaluation session JSONs |
| `add_derived_answers.py` | Compute derived accuracy from VLM multi-vote results | - | Augmented evaluation JSONs |
| `download_and_process_vlm_results.py` | Download VLM results from Modal volumes | - | Local evaluation JSONs |

**Evaluation commands** (examples):

```bash
# Evaluate VLM on test set (5 votes per sample)
modal run scripts/evaluation/modal_proofreading_inference.py::evaluate \
    --task merge_action --checkpoint merge_action_lora_merged --votes 5

# Cross-species linear probe evaluation
modal run scripts/evaluation/modal_linear_probe_cross_species.py::batch_evaluate

# ResNet calibration
python scripts/evaluation/compute_resnet_calibration.py --analyze-all

# Human evaluation GUI
python scripts/evaluation/human_eval.py --user-id evaluator1 --num-samples 32
```

**External model evaluation** (GPT-5, Gemini-3-Pro): These were evaluated using the same `proofreading_evaluator.py` pipeline with API-based inference rather than local VLM serving. The evaluation JSONs in `data/final_data/` contain the raw predictions.

### Stage 3: Data Compilation

Compilation scripts in `scripts/compilation/` aggregate raw evaluation outputs into summary JSONs.

| Script | Input | Output |
|---|---|---|
| `compile_resnet_results.py` | `data/resnet_evaluations/*.json` | `data/final_data/resnet_results.json` |
| `compile_resnet_cross_species.py` | `data/resnet_evaluations/*.json` | `data/final_data/resnet_cross_species_results.json` |
| `compile_calibration_results.py` | `data/calibration_results_cross_species/` | `data/final_data/calibration_cross_species_results.json` |
| `compute_vlm_calibration.py` | VLM evaluation JSONs | `data/final_data/vlm_calibration_all_results.json` |
| `update_calibration_with_vlm.py` | VLM calibration data | Updated calibration results |
| `extract_model_metadata.py` | Hardcoded constants | `data/model_metadata.json` |

These scripts are lightweight (no GPU required) and can be re-run locally. Note: they currently reference original repo paths and would need path updates to run within this reproduction directory.

### Stage 4: Figure Generation (self-contained)

Plotting scripts in `scripts/plotting/` are fully self-contained. They only depend on `matplotlib`, `numpy`, and the JSON data files in `data/`.

```bash
# All figures at once
bash generate_all_figures.sh

# Or individually:
python scripts/plotting/plot_figure2_scaling_curves.py
python scripts/plotting/plot_figure3a_subplots.py
python scripts/plotting/plot_figure3b_ece_vs_data.py
python scripts/plotting/plot_figure3c_calibration.py
python scripts/plotting/generate_benchmark_table.py
```

---

## Directory Structure

```
reproduction/
├── README.md                           # This file
├── requirements.txt                    # matplotlib, numpy (for figure generation)
├── generate_all_figures.sh             # One-command figure runner
│
├── scripts/
│   ├── training/                       # Stage 1: Model training (Modal GPU)
│   │   ├── modal_resnet_finetune.py    #   ResNet-50 training (frozen/finetuned/small)
│   │   ├── modal_qwen_finetune.py      #   VLM finetuning + linear probe sweeps
│   │   ├── run_scaling_experiments.sh  #   Parallel training orchestration
│   │   ├── trajectory_task_config.py   #   Task config for training
│   │   └── upload_utils.py             #   Model upload utilities
│   │
│   ├── evaluation/                     # Stage 2: Model evaluation
│   │   ├── modal_proofreading_inference.py  # VLM inference backend (vLLM)
│   │   ├── proofreading_evaluator.py   #   End-to-end evaluation pipeline
│   │   ├── modal_linear_probe_cross_species.py  # Cross-species linear probe
│   │   ├── evaluate_cross_species.py   #   Cross-species ResNet evaluation
│   │   ├── evaluate_linear_probe_cross_species.py  # LP cross-species eval
│   │   ├── compute_resnet_calibration.py  # Calibration metrics (ECE, MCE)
│   │   ├── human_eval.py              #   Human baseline collection (Gradio)
│   │   ├── add_derived_answers.py     #   Derive answers from multi-vote
│   │   └── download_and_process_vlm_results.py  # Download from Modal
│   │
│   ├── compilation/                    # Stage 3: Data aggregation
│   │   ├── compile_resnet_results.py
│   │   ├── compile_resnet_cross_species.py
│   │   ├── compile_calibration_results.py
│   │   ├── compute_vlm_calibration.py
│   │   ├── update_calibration_with_vlm.py
│   │   └── extract_model_metadata.py
│   │
│   └── plotting/                       # Stage 4: Figure generation
│       ├── plot_figure2_scaling_curves.py
│       ├── plot_figure3a_subplots.py
│       ├── plot_figure3b_ece_vs_data.py
│       ├── plot_figure3c_calibration.py
│       └── generate_benchmark_table.py
│
├── src/                                # Core library modules
│   ├── environment/
│   │   ├── task_configs.py             # Task definitions (all 4 benchmark tasks)
│   │   ├── problem_spec.py             # Problem specification dataclass
│   │   └── env_utils.py                # Environment utilities
│   └── training/
│       └── question_dataset.py         # PyTorch Dataset for training
│
├── config/
│   └── proofreading_models.json        # Model checkpoint paths and configs
│
├── data/                               # Pre-computed evaluation results
│   ├── linear_probe_sweep/             # Linear probe scaling data (4 tasks)
│   ├── final_data/                     # Compiled evaluation results
│   │   ├── resnet_results.json
│   │   ├── resnet_cross_species_results.json
│   │   ├── calibration_cross_species_results.json
│   │   ├── vlm_calibration_all_results.json
│   │   ├── vlm_evaluations/            # VLM raw evaluation files
│   │   └── vlm_evaluations_with_derived/
│   ├── human_eval/                     # Human baseline evaluation sessions
│   ├── linear_probe_cross_species/     # Cross-species linear probe summary
│   ├── calibration_results_all/        # Calibration summary across checkpoints
│   ├── resnet_evaluations/             # Raw per-checkpoint ResNet results
│   ├── calibration_results_cross_species/  # Raw cross-species calibration
│   └── model_metadata.json             # Model parameter counts
│
└── output/                             # Generated figures
    └── .gitkeep
```

## Infrastructure Requirements

| Stage | Requirements |
|---|---|
| Figure generation only | Python 3.8+, matplotlib, numpy |
| Data compilation | Python 3.8+, numpy |
| Model evaluation | Modal account, GPU (A10G+), torch, transformers, scikit-learn |
| Model training | Modal account, GPU (A100/H100), torch, transformers, peft, vllm |
| Training data generation | caveclient (MICrONS dataset access), kaolin (GPU rendering) |
