#!/bin/bash
# Run ResNet/ViT scaling experiments across models, tasks, and tile/image sizes.
#
# ResNet uses --tile-size (preserves per-tile resolution, matching VLM input).
# ViT uses --image-size (ViT attention is quadratic in patches; rectangular
#   tile grids would exceed A10G VRAM at any batch size).
#
# Batch sizes are tuned per (model, task, size) based on grid dimensions:
#   3-image tasks (1x3 grid): smaller grids, larger batches OK
#   6-image tasks (2x3 grid): ~4x more pixels, needs smaller batches
#
# Launches up to 10 in parallel (Modal GPU concurrency limit).

SCRIPT="scripts/model-post-training/modal_resnet_finetune.py::main"
MAX_PARALLEL=5
LOG_DIR="logs/scaling_experiments"
mkdir -p "$LOG_DIR"

# Task -> num_images (determines grid size and memory)
# merge_action:                     3 imgs -> 1x3 grid
# split_action:                     3 imgs -> 1x3 grid
# merge_error_identification:       3 imgs -> 1x3 grid
# endpoint_error_id_with_em:        6 imgs -> 2x3 grid (~4x pixels)
TASKS=("merge_action" "split_action" "merge_error_identification" "endpoint_error_identification_with_em")
SIZES=(384 512)
MODELS=("resnet50" "vit_b_16" "vit_l_16")

get_size_flag() {
    local model=$1
    if [[ "$model" == resnet* ]]; then
        echo "tile-size"
    else
        echo "image-size"
    fi
}

get_batch_size() {
    local model=$1
    local task=$2
    local size=$3

    # 6-image tasks have ~4x the pixels of 3-image tasks
    local is_big_grid=false
    if [[ "$task" == "endpoint_error_identification_with_em" ]]; then
        is_big_grid=true
    fi

    if [[ "$model" == "resnet50" ]]; then
        if $is_big_grid; then
            # 2x3 grid: tile 384 → 768x1152, tile 512 → 1024x1536
            if [[ "$size" == "512" ]]; then echo 4
            else echo 8; fi
        else
            # 1x3 grid: tile 384 → 384x1152, tile 512 → 512x1536
            if [[ "$size" == "512" ]]; then echo 16
            else echo 32; fi
        fi
    elif [[ "$model" == "vit_b_16" ]]; then
        # ViT uses --image-size (square), so task grid doesn't matter as much
        if [[ "$size" == "512" ]]; then echo 32
        else echo 48; fi
    elif [[ "$model" == "vit_l_16" ]]; then
        if [[ "$size" == "512" ]]; then echo 8
        else echo 16; fi
    else
        echo 16
    fi
}

TOTAL=$((${#MODELS[@]} * ${#TASKS[@]} * ${#SIZES[@]}))
COUNT=0
PIDS=()
LABELS=()
LOGFILES=()

echo "=========================================="
echo "Starting scaling experiments: $TOTAL runs (max $MAX_PARALLEL parallel)"
echo "Models: ${MODELS[*]}"
echo "Tasks: ${TASKS[*]}"
echo "Sizes: ${SIZES[*]}"
echo "ResNet: --tile-size | ViT: --image-size"
echo "Logs: $LOG_DIR/"
echo "=========================================="

wait_for_slot() {
    while true; do
        local running=0
        local new_pids=()
        local new_labels=()
        local new_logfiles=()
        for i in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$i]}" 2>/dev/null; then
                running=$((running + 1))
                new_pids+=("${PIDS[$i]}")
                new_labels+=("${LABELS[$i]}")
                new_logfiles+=("${LOGFILES[$i]}")
            else
                wait "${PIDS[$i]}" 2>/dev/null
                local exit_code=$?
                if [[ $exit_code -eq 0 ]]; then
                    echo "  DONE: ${LABELS[$i]}"
                else
                    echo "  FAIL: ${LABELS[$i]} (exit $exit_code) -- see ${LOGFILES[$i]}"
                fi
            fi
        done
        PIDS=("${new_pids[@]}")
        LABELS=("${new_labels[@]}")
        LOGFILES=("${new_logfiles[@]}")
        if [[ $running -lt $MAX_PARALLEL ]]; then
            break
        fi
        sleep 5
    done
}

for model in "${MODELS[@]}"; do
    size_flag=$(get_size_flag "$model")
    for task in "${TASKS[@]}"; do
        for size in "${SIZES[@]}"; do
            COUNT=$((COUNT + 1))
            batch_size=$(get_batch_size "$model" "$task" "$size")
            label="${model}_${task}_${size_flag}${size}"
            logfile="$LOG_DIR/${label}.log"

            wait_for_slot

            echo "[$COUNT/$TOTAL] Launching: $label (--${size_flag} $size, bs=$batch_size)"

            modal run "$SCRIPT" \
                --model "$model" \
                --task "$task" \
                --"$size_flag" "$size" \
                --batch-size "$batch_size" \
                --max-steps 10000 \
                --use-wandb \
                --class-balance \
                > "$logfile" 2>&1 &

            PIDS+=($!)
            LABELS+=("$label")
            LOGFILES+=("$logfile")
        done
    done
done

echo ""
echo "All $TOTAL jobs launched. Waiting for remaining to finish..."

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" 2>/dev/null
    exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "  DONE: ${LABELS[$i]}"
    else
        echo "  FAIL: ${LABELS[$i]} (exit $exit_code) -- see ${LOGFILES[$i]}"
    fi
done

echo ""
echo "=========================================="
echo "All $TOTAL experiments finished."
echo "Logs in: $LOG_DIR/"
echo "=========================================="
