export CUDA_VISIBLE_DEVICES=6

python /root/workspace/agent-rm/Agent-Verifier/src/score_vanilla_infer.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_vanilla.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/qwen3-4b_math-500-haswrong104_bon-128.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/0919/qwen3-4b_math-500-haswrong104_bon-128_vanilla_score1_by_qwen3-4b.jsonl \
  --record-batch-size 1 \
  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
