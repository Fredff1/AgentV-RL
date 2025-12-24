export CUDA_VISIBLE_DEVICES=4,6


bash run_refine_entry.sh \
  --candidate-config config/cand.yaml \
  --verifier-config  config/ver.yaml \
  --input data/refine.jsonl \
  --output out/final_forward.jsonl \
  --round-output-dir out/rounds_forward \
  --metrics-output-dir out/metrics_forward \
  --exp-name exp_forward \
  --verifier-type forward \
  --candidate-model-path \
  --verifier-model-path \
  --num-candidate-workers 1 \
  --num-verifier-workers 1 \
  --batch-size 64 \
  --max-refine-rounds 128 \
  --thinking-candidate \
  ----thinking-verifier \




