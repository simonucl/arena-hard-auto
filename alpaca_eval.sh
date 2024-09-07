CHECKPOINT_PATHS=(
    meta-llama/Meta-Llama-3.1-8B-Instruct
)
PENALTYS=(1.0 1.15 1.3)
NUM_GPUS=1
mkdir -p results/alpaca_eval

# for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
for PENALTY in "${PENALTYS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    # Step 1: Generate config
    python3 -m eval.alpaca_eval.gen \
        --model_name_or_path $CHECKPOINT_PATH \
        --save_dir results/alpaca_eval/${MODEL_NAME}_${PENALTY} \
        --max_new_tokens 4096 \
        --use_vllm \
        --repetition_penalty $PENALTY
        
    alpaca_eval --model_outputs results/alpaca_eval/${MODEL_NAME}_${PENALTY}/${MODEL_NAME}-greedy-long-output.json
done

python3 show_result.py