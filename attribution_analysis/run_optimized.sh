#!/bin/bash

# Optimized Attribution Analysis Script
# Uses multi-alpha batching and mixed precision for 5-30x speedup

# NRMS Model - 2000 steps (OPTIMIZED - ~20-40 minutes instead of hours)
echo "Running NRMS attribution analysis with optimizations..."
python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 10000 \
    --n_steps 2000 \
    --top_k 20 \
    --top_k_sample 20 \
    --use_optimized \
    --use_amp \
    --alpha_batch_size 10

# # NAML Model - 200 steps (OPTIMIZED - ~10-20 minutes instead of hours)
# echo "Running NAML attribution analysis with optimizations..."
# python analyze_attributions.py \
#     --config ../configs/attribution/naml_bert_finetune.yaml \
#     --dataset benchmark \
#     --n_samples 10000 \
#     --n_steps 200 \
#     --top_k 20 \
#     --top_k_sample 20 \
#     --use_optimized \
#     --use_amp \
#     --alpha_batch_size 10

# For maximum speed (requires more GPU memory ~8-12GB):
# --alpha_batch_size 50

# For lower GPU memory (~4-6GB):
# --alpha_batch_size 10

# For minimal GPU memory (~3-4GB):
# --alpha_batch_size 5

# To disable optimizations (original implementation):
# --no_optimized --no_amp
