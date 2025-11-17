python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset benchmark \
    --n_samples 10000 \
    --n_steps 50 \
    --top_k 15

python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset train_clean \
    --n_samples 10000 \
    --n_steps 50 \
    --top_k 15

python analyze_attributions.py \
    --config ../configs/attribution/nrms_bert_finetune.yaml \
    --dataset train_poisoned \
    --n_samples 10000 \
    --n_steps 50 \
    --top_k 15