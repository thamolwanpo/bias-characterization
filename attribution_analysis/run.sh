# Train DIFFMASK and analyze
python analyze_diffmask.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --n_samples 100 \
    --n_epochs 10 \
    --probe_type simple \
    --constraint_margin 0.1 \
    --top_k 15

# python analyze_attributions.py \
#     --config ../configs/attribution/nrms_bert_finetune.yaml \
#     --dataset benchmark \
#     --n_samples 40000 \
#     --n_steps 50 \
#     --top_k 15

# python analyze_attributions.py \
#     --config ../configs/attribution/nrms_bert_finetune.yaml \
#     --dataset train_clean \
#     --n_samples 40000 \
#     --n_steps 50 \
#     --top_k 15

# python analyze_attributions.py \
#     --config ../configs/attribution/nrms_bert_finetune.yaml \
#     --dataset train_poisoned \
#     --n_samples 40000 \
#     --n_steps 50 \
#     --top_k 15