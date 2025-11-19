python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 10000 \
    --n_steps 1000 \
    --top_k 20 \
    --top_k_sample 20

python analyze_attributions.py \
    --config ../configs/attribution/naml_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 10000 \
    --n_steps 200 \
    --top_k 20 \
    --top_k_sample 20