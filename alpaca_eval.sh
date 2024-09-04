CHECKPOINT_PATHS=(
    simonycl/llama-3.1-8b-instruct-single-judge
    simonycl/llama-3.1-8b-instruct-agg-judge
    simonycl/llama-3.1-8b-instruct-armorm
)

NUM_GPUS=1
mkdir -p results/alpaca_eval

for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # Step 1: Generate config
    python3 -m eval.alpaca_eval.gen \
        --model_name_or_path $CHECKPOINT_PATH \
        --save_dir results/alpaca_eval/${MODEL_NAME} \
        --eval_batch_size 16 \
        --max_new_tokens 4096

    alpaca_eval --model_outputs results/alpaca_eval/${MODEL_NAME}/$MODEL_NAME-greedy-long-output.json
done

python3 show_result.py