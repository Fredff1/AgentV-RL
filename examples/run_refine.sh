export CUDA_VISIBLE_DEVICES=2,3

SRC_DIR="Agentic-Verfifier/src"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"
echo "PYTHONPATH = $PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

EXP_NAME=
WORKING_DIR=

bash examples/run_refine_entry.sh \
  --candidate-config config/refine_candidate.yaml \
  --verifier-config  config/refine_verifier.yaml \
  --input aime.jsonl \
  --output ${WORKING_DIR}/${EXP_NAME}/last_round.jsonl \
  --round-output-dir ${WORKING_DIR}/${EXP_NAME} \
  --metrics-output-dir ${WORKING_DIR}/${EXP_NAME}/metrics \
  --exp-name ${EXP_NAME} \
  --verifier-type forward \
  --candidate-model-path Path-to-candidate\
  --verifier-model-path Path-to-verifier \
  --num-candidate-workers 2 \
  --num-verifier-workers 2 \
  --batch-size 16 \
  --max-refine-rounds 128 \
  --thinking-candidate \
  --thinking-verifier \