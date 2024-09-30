CHECKPOINT_PATH=$1
GPU_ID=$2
MODEL_NAME=$(basename $CHECKPOINT_PATH)

# Step 1: Generate config
python3 gen_config.py \
    --model_path $CHECKPOINT_PATH

# Step 3: Run gen answer
CUDA_VISIBLE_DEVICES=$GPU_ID python3 gen_answer_hf.py \
    --setting-file config/$MODEL_NAME/gen_answer_config.yaml \
    --endpoint-file config/$MODEL_NAME/api_config.yaml

python3 show_result.py
