CHECKPOINT_PATHS=(
    # meta-llama/Meta-Llama-3-8B-Instruct
    simonycl/llama-3-8b-instruct-agg-judge
    simonycl/llama-3-8b-instruct-single-judge
)

NUM_GPUS=1
SLEEP=300
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # Step 1: Generate config
    python3 gen_config.py \
        --model_path $CHECKPOINT_PATH

    # Step 2: Start vllm server (TODO: have to be in the background and wait for it to be ready, kill it after eval)
    python3 -m vllm.entrypoints.openai.api_server --model $CHECKPOINT_PATH --dtype auto --api-key token-abc123 --port 8000 --tensor-parallel-size $NUM_GPUS > vllm.log &

    # Wait for the server to be ready
    sleep $SLEEP

    # Step 3: Run gen answer
    python3 gen_answer.py \
        --setting-file config/$MODEL_NAME/gen_answer_config.yaml \
        --endpoint-file config/$MODEL_NAME/api_config.yaml

    # Step 4: Run gen judgement
    python3 gen_judgement.py \
        --setting-file config/$MODEL_NAME/judge_config.yaml \
        --endpoint-file config/$MODEL_NAME/api_config.yaml

    # Step 5: Kill vllm server by port and kill all with name ray
    pkill -f vllm
    pkill -f multiprocessing

done

python3 show_result.py