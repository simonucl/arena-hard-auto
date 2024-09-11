MODEL_NAME_OR_PATH=$1
NUM_GPUS=4

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_NAME_OR_PATH,trust_remote_code=true,data_parallel_size=$NUM_GPUS \
    --tasks ifeval \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size 16 \
    --output_path results || exit 1

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_NAME_OR_PATH,trust_remote_code=true,data_parallel_size=$NUM_GPUS,gpu_memory_utilization=0.5 \
    --tasks leaderboard_gpqa \
    --num_fewshot 0 \
    --apply_chat_template \
    --batch_size auto \
    --output_path results || exit 1

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_NAME_OR_PATH,trust_remote_code=true,data_parallel_size=$NUM_GPUS \
    --tasks gsm8k \
    --num_fewshot 5 \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs temperature=0 \
    --output_path results || exit 1

lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_NAME_OR_PATH,trust_remote_code=true,data_parallel_size=$NUM_GPUS \
    --tasks leaderboard_math_hard \
    --num_fewshot 4 \
    --apply_chat_template \
    --log_samples \
    --gen_kwargs temperature=0 \
    --output_path results || exit 1
