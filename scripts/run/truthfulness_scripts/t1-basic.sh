model_id=$1
dataset_categories=$2

dataset_ids=(
    "d-basic-attribute"
    "d-basic-object"
    "d-basic-scene"
    "g-basic-grounding"
    "g-basic-ocr"
)

for dataset_id in "${dataset_ids[@]}";
do
    clean_dataset_id=${dataset_id#*-}
    PYTHONPATH=.. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 
    python inference.py \
    --follow_model_id="llama-3-2-chat" \
    --reason_model_id="llama-3-2-chat" \
    --dataset_categories="truthfulness" \
    --prompt_config_path="configs/prompt_config.yaml" \
    --task_config_path configs/task/truthfulness/t1-${clean_dataset_id}.yaml \
    --dataset_id=${dataset_id} \
    --log_file="logs/truthfulness/t1-basic/${model_id}/${dataset_id}.json"