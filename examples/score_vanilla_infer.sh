export CUDA_VISIBLE_DEVICES=0,1

python src/score_vanilla_infer.py \
  --config config/score_vanilla.yaml \
  --input  bon_verify.jsonl \
  --output bon_result.jsonl \
  --record-batch-size 1 \
  --append \

