export CUDA_VISIBLE_DEVICES=3,4

bash examples/run_verify_entry.sh \
  --task-name gaokao-2023-eval \
  --exp-name qwen3-4b \
  --num-workers 2 \
  --config config/distrubute_verify.yaml \
  --model-path models/qwen3-4b \
  --input datasets.jsonl \
  --output-dir datasets/gaokao2023 \
  --log-dir ./log \
  --start-idx 0 \
  --enable-thinking
