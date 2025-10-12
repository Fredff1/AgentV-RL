export CUDA_VISIBLE_DEVICES=2,3


python /root/workspace/agent-rm/Agent-Verifier/src/score_tool_complex_agent_multi_head.py \
  --config /root/workspace/agent-rm/Agent-Verifier/config/score_tool_complex.yaml \
  --input  /root/workspace/agent-rm/datasets/math500/bon/qwen2.5-7b-math-bon128-math500-eval-100.jsonl \
  --output /root/workspace/agent-rm/datasets/math500/1011/qwen2.5-7b-math-bon128-math500-eval-100_complex_agent_score1_by_qwen2.5-7b-multihead.jsonl \
  --record-batch-size 1 \
  --include_full_meta \
  --start_idx 0 \
  --append \

  # --judge-system-file /root/workspace/agent-rm/prompts/judge_system.txt \
  # --judge-user-file   /root/workspace/agent-rm/prompts/judge_user.txt
