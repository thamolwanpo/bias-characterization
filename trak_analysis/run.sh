#!/bin/bash

# TRAK Analysis for News Recommendation Models
# Analyzes training sample influence using TRAK scores

# Default configuration
CONFIG="configs/trak/naml_bert_finetune.yaml"
DATASET_TYPE="train_clean"
MODEL_TYPE="clean"
TOP_K=100
PROJ_DIM=512

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset_type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --proj_dim)
            PROJ_DIM="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "TRAK Analysis"
echo "========================================"
echo "Config: $CONFIG"
echo "Dataset: $DATASET_TYPE"
echo "Model: $MODEL_TYPE"
echo "Top-K: $TOP_K"
echo "Projection Dim: $PROJ_DIM"
echo "========================================"
echo ""

# Run TRAK analysis
python trak_analysis/analyze_trak.py \
    --config "$CONFIG" \
    --dataset_type "$DATASET_TYPE" \
    --model_type "$MODEL_TYPE" \
    --top_k "$TOP_K" \
    --proj_dim "$PROJ_DIM"

echo ""
echo "========================================"
echo "Analysis Complete!"
echo "========================================"
