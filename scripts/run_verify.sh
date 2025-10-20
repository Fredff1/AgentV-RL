export CUDA_VISIBLE_DEVICES=3,6,7

NUM_WORKERS=3

python /root/workspace/agent-rm/Agent-Verifier/src/run_verify.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/distrubute_verify.yaml \
  --input  /root/workspace/agent-rm/datasets/gaokao2023/bon/qwen2.5-7b-math-bon128-gaokao-2023.jsonl \
  --output /root/workspace/agent-rm/datasets/gaokao2023/1021/qwen2.5-7b-math-bon128-gaokao-2023-eval-1_by_qwen2.5-7b.jsonl \
  --model_path /root/workspace/agent-rm/models/Qwen-2.5-7B-Instruct \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx 0 \
  --append \
  --num-workers $NUM_WORKERS \
  --max-inflight-batches $NUM_WORKERS \

  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
