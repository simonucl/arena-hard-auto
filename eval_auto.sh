CHECKPOINT_PATHS=(
    # /mnt/nfs/public/hf/models/meta-llama/Meta-Llama-3.1-8B-Instruct
    simonycl/llama-3-8b-instruct-armorm-judge
    simonycl/llama-3-8b-instruct-single-judge
    simonycl/llama-3-8b-instruct-agg-judge
    simonycl/llama-3-8b-instruct-metamath-single-judge
    simonycl/llama-3-8b-instruct-metamath-armorm
    simonycl/llama-3-8b-instruct-metamath-agg-judge
    simonycl/llama-3.1-8b-instruct-armorm-iter0
    simonycl/llama-3.1-8b-instruct-armorm-iter1
    simonycl/llama-3.1-8b-instruct-armorm-judge-iter2
    simonycl/llama-3.1-8b-instruct-armorm-judge-iter3
)

for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    MODEL_NAME=$(basename $CHECKPOINT_PATH)
    echo "Evaluating $MODEL_NAME"
    bash scripts/eval.sh $CHECKPOINT_PATH
done