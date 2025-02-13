# dev: PYTHONPATH=. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python inference.py
from dataclasses import dataclass
import tqdm
import transformers
import omegaconf
from tasks.base import BaseTask
import pdb

@dataclass
class ModelArguments:
    follow_model_id:   str = "llama-3-2-chat"
    reason_model_id:   str = "llama-3-2-chat"
    load_in_4bit:       bool = False
    use_flash_attention: bool = True

@dataclass
class TaskArguments:
    prompt_config_path:   str = "configs/prompt_config.yaml"
    task_config_path:   str = ""
    follow_rules: bool = True
    dataset_categories: str = "truthfulness"
    dataset_id: str = "d-basic-attribute"
    log_file: str = ""
    long_pipe: bool = True
@dataclass
class InferenceArguments:
    output_dir:     str = "results/model:llama-3-2-chat/task:mm-safety_bench/categories:13-Gov_Decision"


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, TaskArguments, InferenceArguments))
    model_args, task_args, inference_args = parser.parse_args_into_dataclasses()

    prompt_cfg = omegaconf.OmegaConf.load(task_args.prompt_config_path)
    task_cfg = omegaconf.OmegaConf.load(task_args.task_config_path)
    safety_rules = prompt_cfg.tasks.specific_safety_rules[task_args.dataset_categories]
    generation_kwargs = task_cfg.generation_kwargs
    evaluator_seq_cfgs = task_cfg.evaluator_seq_cfgs
    runner = BaseTask(dataset_id=task_args.dataset_id,follow_model_id=model_args.follow_model_id,reason_model_id=model_args.reason_model_id,follow_rules=task_args.follow_rules,safety_rules=safety_rules,dataset_categories=task_args.dataset_categories,generation_kwargs=generation_kwargs, log_file=task_args.log_file, evaluator_seq_cfgs=evaluator_seq_cfgs)
    
    runner.pipeline() if task_args.long_pipe else runner.pipeline_ref()
    
if __name__ == "__main__":
    inference()