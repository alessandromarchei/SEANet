#!/usr/bin/env bash

set -uo pipefail


# ============================================================
# Test all saved validation checkpoints
#
# Usage:
#
#   bash scripts/test_checkpoints.sh \
#       runs/SEANet_12.5fps_retrain_1/model/ \
#       --data_list configs/data_list.csv \
#       --audio_path /path/to/test/audio \
#       --visual_path /path/to/visual \
#       --visual_fps 12.5
#
# Additional arguments are forwarded directly to eval.py.
# ============================================================


# ============================================================
# Arguments
# ============================================================

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo
    echo "  $0 CHECKPOINT_DIR [eval.py arguments...]"
    echo
    echo "Example:"
    echo
    echo "  $0 runs/SEANet_12.5fps_retrain_1/model/ \\"
    echo "      --data_list configs/data_list.csv \\"
    echo "      --audio_path /path/to/audio \\"
    echo "      --visual_path /path/to/visual \\"
    echo "      --visual_fps 12.5"
    echo
    exit 1
fi


CHECKPOINT_DIR="$1"
shift

EXTRA_ARGS=("$@")


# ============================================================
# Configuration
# ============================================================

EVAL_SCRIPT="eval.py"

# All evaluation directories are created here:
#
#   runs/eval_val_epoch_002/
#   runs/eval_val_epoch_004/
#   ...
#
EVAL_ROOT="runs"

# Global summary
MASTER_LOG="${EVAL_ROOT}/checkpoint_test_summary.log"


# ============================================================
# Sanity checks
# ============================================================

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint directory does not exist:"
    echo "  $CHECKPOINT_DIR"
    exit 1
fi


if [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "ERROR: eval.py not found:"
    echo "  $EVAL_SCRIPT"
    echo
    echo "Run this script from the SEANet repository root."
    exit 1
fi


mkdir -p "$EVAL_ROOT"


# ============================================================
# Find validation checkpoints
# ============================================================

mapfile -t CHECKPOINTS < <(
    find "$CHECKPOINT_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'val_epoch_*.pt' \
        -print \
    | sort -V
)


NUM_CHECKPOINTS="${#CHECKPOINTS[@]}"


if [[ "$NUM_CHECKPOINTS" -eq 0 ]]; then
    echo "ERROR: no validation checkpoints found."
    echo
    echo "Expected files like:"
    echo
    echo "  val_epoch_002.pt"
    echo "  val_epoch_004.pt"
    echo "  val_epoch_006.pt"
    echo
    echo "Inside:"
    echo "  $CHECKPOINT_DIR"
    exit 1
fi


# ============================================================
# Header
# ============================================================

{
    echo "============================================================"
    echo "TEST EVALUATION OF VALIDATION CHECKPOINTS"
    echo "============================================================"
    echo
    echo "Checkpoint directory : $CHECKPOINT_DIR"
    echo "Checkpoints found    : $NUM_CHECKPOINTS"
    echo "Eval script          : $EVAL_SCRIPT"
    echo "Evaluation root      : $EVAL_ROOT"
    echo
    echo "Checkpoints:"
    
    for CKPT in "${CHECKPOINTS[@]}"; do
        echo "  $(basename "$CKPT")"
    done

    echo
} | tee "$MASTER_LOG"


# ============================================================
# Counters
# ============================================================

SUCCESS=0
FAILED=0


# ============================================================
# Evaluate checkpoints
# ============================================================

for i in "${!CHECKPOINTS[@]}"; do

    CKPT="${CHECKPOINTS[$i]}"

    CURRENT=$((i + 1))

    # --------------------------------------------------------
    # Checkpoint name
    #
    # val_epoch_002.pt
    #       ↓
    # val_epoch_002
    # --------------------------------------------------------

    NAME="$(basename "$CKPT" .pt)"


    # --------------------------------------------------------
    # Output directory
    #
    # val_epoch_002
    #       ↓
    # runs/eval_val_epoch_002/
    # --------------------------------------------------------

    OUTPUT_DIR="${EVAL_ROOT}/eval_${NAME}"

    LOG_FILE="${OUTPUT_DIR}/eval.log"


    mkdir -p "$OUTPUT_DIR"


    # --------------------------------------------------------
    # Console header
    # --------------------------------------------------------

    echo
    echo "============================================================"
    echo "[$CURRENT/$NUM_CHECKPOINTS] $NAME"
    echo "============================================================"
    echo
    echo "Checkpoint : $CKPT"
    echo "Output     : $OUTPUT_DIR"
    echo "Log        : $LOG_FILE"
    echo


    # --------------------------------------------------------
    # Master log header
    # --------------------------------------------------------

    {
        echo
        echo "============================================================"
        echo "[$CURRENT/$NUM_CHECKPOINTS] $NAME"
        echo "============================================================"
        echo "Checkpoint: $CKPT"
        echo "Output:     $OUTPUT_DIR"
        echo
    } >> "$MASTER_LOG"


    # --------------------------------------------------------
    # Evaluation
    #
    # Fixed:
    #   backbone   = seanet
    #   split      = test
    #   save_audio = 0
    #
    # Everything else is forwarded from the command line.
    # --------------------------------------------------------

    python "$EVAL_SCRIPT" \
        --model "$CKPT" \
        --backbone seanet \
        --split test \
        --save_audio 0 \
        --output_dir "$OUTPUT_DIR" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG_FILE"


    # IMPORTANT:
    #
    # Because output is piped through tee, $? would give the
    # exit code of tee. PIPESTATUS[0] gives the exit code of
    # python instead.
    STATUS=${PIPESTATUS[0]}


    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    if [[ "$STATUS" -eq 0 ]]; then

        SUCCESS=$((SUCCESS + 1))

        echo
        echo "[OK] $NAME"


        {
            echo
            echo "STATUS: OK"
            echo

            # Extract useful final metrics from the log.
            grep -E \
                'SI-SDR|SISDR|SDR|SI-SDRi|SISDRi|SDRi' \
                "$LOG_FILE" \
                | tail -n 20 || true

        } >> "$MASTER_LOG"


    # --------------------------------------------------------
    # Failure
    # --------------------------------------------------------

    else

        FAILED=$((FAILED + 1))

        echo
        echo "[FAILED] $NAME"
        echo "Exit code: $STATUS"


        {
            echo
            echo "STATUS: FAILED"
            echo "Exit code: $STATUS"
        } >> "$MASTER_LOG"

    fi

done


# ============================================================
# Final summary
# ============================================================

{
    echo
    echo "============================================================"
    echo "EVALUATION COMPLETE"
    echo "============================================================"
    echo
    echo "Total checkpoints : $NUM_CHECKPOINTS"
    echo "Successful        : $SUCCESS"
    echo "Failed            : $FAILED"
    echo
    echo "Evaluation directories:"
    echo "  ${EVAL_ROOT}/eval_val_epoch_XXX/"
    echo
    echo "Summary:"
    echo "  $MASTER_LOG"
    echo

} | tee -a "$MASTER_LOG"


# ============================================================
# Exit status
# ============================================================

if [[ "$FAILED" -gt 0 ]]; then
    exit 2
fi

exit 0