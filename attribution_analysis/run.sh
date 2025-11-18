python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 40000 \
    --n_steps 50 \
    --top_k 15 \
    --top_k_sample 10

python analyze_attributions.py \
    --config ../configs/attribution/naml_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 40000 \
    --n_steps 50 \
    --top_k 15 \
    --top_k_sample 10