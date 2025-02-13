#srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 python download.py

from huggingface_hub import snapshot_download
from huggingface_hub import login

login(token="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")  # Replace with your actual token

# For the whole dataset
snapshot_download(repo_id="thu-ml/MultiTrust", local_dir="./Multitrust", repo_type="dataset", allow_patterns=['*'])
