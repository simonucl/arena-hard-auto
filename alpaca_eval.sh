CHECKPOINT_PATHS=(
    meta-llama/Meta-Llama-3-8B-Instruct
)

NUM_GPUS=1
mkdir -p results/alpaca_eval

for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # Step 1: Generate config
    python3 -m eval.alpaca_eval.gen \
        --model_path $CHECKPOINT_PATH
        --save_dir results/alpaca_eval/${MODEL_NAME}
        --max_new_tokens 2048

    alpaca_eval --model_outputs results/alpaca_eval/${MODEL_NAME}/$MODEL_NAME-greedy-long-output.json
done

python3 show_result.py