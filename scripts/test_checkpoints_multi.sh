#!/bin/bash

# Set GPU (default to GPU 0, can be changed via command line argument)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dfot

# Base directory for checkpoints
CHECKPOINT_DIR="./outputs/20260101_154317_vae_wan_1d_multi_272"
CONFIG_FILE="configs/vae_wan_1d_multi_272_test.yaml"

# Test identifier (used in save_dir and log_dir)
TEST_ID="multi_272_4_0_0"

# Custom save directory for test results
CUSTOM_SAVE_DIR="./outputs/test_results_${TEST_ID}"
mkdir -p "${CUSTOM_SAVE_DIR}"

# Log directory
LOG_DIR="./outputs/test_logs/${TEST_ID}"
mkdir -p "${LOG_DIR}"

# Checkpoints to test (from 100000 to 150000)
CHECKPOINTS=(160000 170000 180000 190000 200000 210000 220000 230000 240000 250000 260000 270000 280000 290000 300000)

# Main log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/test_checkpoints_${TIMESTAMP}.log"

echo "==========================================" | tee -a "${MAIN_LOG}"
echo "Starting checkpoint testing on GPU ${GPU_ID}" | tee -a "${MAIN_LOG}"
echo "Log file: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

# Loop through each checkpoint
for step in "${CHECKPOINTS[@]}"; do
    ckpt_path="${CHECKPOINT_DIR}/step_step=${step}.ckpt"
    step_log="${LOG_DIR}/test_step_${step}_${TIMESTAMP}.log"
    
    echo "" | tee -a "${MAIN_LOG}"
    echo "==========================================" | tee -a "${MAIN_LOG}"
    echo "Testing checkpoint: step_step=${step}.ckpt" | tee -a "${MAIN_LOG}"
    echo "Step log: ${step_log}" | tee -a "${MAIN_LOG}"
    echo "==========================================" | tee -a "${MAIN_LOG}"
    
    # Run test and save to both step log and main log
    python train_vae.py \
        --config "${CONFIG_FILE}" \
        --override "test_ckpt=${ckpt_path}" \
        --override "save_dir=${CUSTOM_SAVE_DIR}" \
        2>&1 | tee -a "${step_log}" | tee -a "${MAIN_LOG}"
    
    echo "Completed testing checkpoint: step_step=${step}.ckpt" | tee -a "${MAIN_LOG}"
done

echo "" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"
echo "All checkpoint tests completed!" | tee -a "${MAIN_LOG}"
echo "Main log saved to: ${MAIN_LOG}" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

