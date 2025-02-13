# srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 bash scripts/run/safety_scripts/s4-typographic-jailbreaking.sh llava-v1.5-7b

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "typographic-prompt-and-behavior"
    "typographic-prompt"
    "typographic-behavior"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/s4-s5-s6-jailbreaking.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/safety/s4-typographic-jailbreaking/${model_id}/${dataset_id}.json"
done



