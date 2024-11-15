CHECKPOINT_PATHS=(
    /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3.1-8B-Instruct
)

EVAL_SET="alpaca_eval" # either "alpaca_eval" or "arena-hard"

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local api_key=token-abc123
  timeout 1200 bash -c '
    until curl -X POST -H "Authorization: Bearer '"$api_key"'" localhost:8000/v1/completions; do
      sleep 1
    done' && return 0 || return 1
}

NUM_GPUS=2
INFERENCE=sglang
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # Step 1: Generate config
    python3 gen_config.py \
        --model_path $CHECKPOINT_PATH

    # Step 2: Start vllm/sglang server
    if [ "$INFERENCE" == "vllm" ]; then
        python3 -m vllm.entrypoints.openai.api_server --model $CHECKPOINT_PATH --dtype auto --api-key token-abc123 --port 8000 --tensor-parallel-size $NUM_GPUS > vllm.log 2>&1 &    
    elif [ "$INFERENCE" == "sglang" ]; then
        python3 -m sglang.launch_server --model-path $CHECKPOINT_PATH --api-key token-abc123 --port 8000 --dp $NUM_GPUS > sglang.log &
    fi

    # Wait for the server to be ready
    if ! wait_for_server; then
        echo "VLLM server failed to start or crashed. Exiting."
        pkill -f multiprocessing
        pkill -f $INFERENCE
        exit 1
    echo "VLLM server started successfully for $MODEL_NAME"
    fi

    # Step 3: Run gen answer
    if [ "$EVAL_SET" == "alpaca_eval" ]; then
        python3 -m eval.alpaca_eval.gen \
            --model_name_or_path $CHECKPOINT_PATH \
            --save_dir results/alpaca_eval/${MODEL_NAME} \
            --eval_batch_size 16 \
            --max_new_tokens 4096 \
            --use_vllm_server

        alpaca_eval --model_outputs results/alpaca_eval/${MODEL_NAME}/${MODEL_NAME}-greedy-long-output.json

    elif [ "$EVAL_SET" == "arena-hard" ]; then

        python3 gen_answer.py \
            --setting-file config/$MODEL_NAME/gen_answer_config.yaml \
            --endpoint-file config/$MODEL_NAME/api_config.yaml

        # Step 4: Run gen judgement
        python3 gen_judgement.py \
            --setting-file config/$MODEL_NAME/judge_config.yaml \
            --endpoint-file config/$MODEL_NAME/api_config.yaml
    fi

    # Step 5: Kill vllm server by port and kill all with name ray
    pkill -f multiprocessing
    pkill -f $INFERENCE
done

python3 show_result.py
