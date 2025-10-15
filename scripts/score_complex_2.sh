export CUDA_VISIBLE_DEVICES=2,3

python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_complex_agent.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_tool_complex.yaml \
  --input  /root/workspace/agent-rm/datasets/gaokao2023/bon/qwen2.5-7b-math-bon128-gaokao-2023.jsonl \
  --output /root/workspace/agent-rm/datasets/gaokao2023/1011/qwen2.5-7b-math-bon128-gaokao-2023-eval_complex_agent_score1_by_qwen3-4b-grpo1009-1.jsonl \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx 0 \
  --append \

  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
