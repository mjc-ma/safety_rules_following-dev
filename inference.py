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
    task_config_path:   str = "configs/task_config.yaml"
    dataset_id: str = "mm-safety-bench"
    follow_rules: bool = True
    dataset_categories: str = "01-Illegal_Activity"

@dataclass
class InferenceArguments:
    output_dir:     str = "results/model:llama-3-2-chat/task:mm-safety_bench/categories:01-Illegal_Activitiy"


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, TaskArguments, InferenceArguments))
    model_args, task_args, inference_args = parser.parse_args_into_dataclasses()

    task_cfg = omegaconf.OmegaConf.load(task_args.task_config_path)
    safety_rules = task_cfg.tasks.safety_rules
    generation_kwargs = task_cfg.generation_kwargs
    log_file = task_cfg.log_file
    evaluator_seq_cfgs = task_cfg.evaluator_seq_cfgs
    # breakpoint()
    runner = BaseTask(dataset_id=task_args.dataset_id,follow_model_id=model_args.follow_model_id,reason_model_id=model_args.reason_model_id,follow_rules=task_args.follow_rules,safety_rules=safety_rules,dataset_categories=task_args.dataset_categories,generation_kwargs=generation_kwargs, log_file=log_file, evaluator_seq_cfgs=evaluator_seq_cfgs)
    #for long_pipeline
    #runner.pipeline()
    #for short_pipeline
    runner.pipeline_ref()
    
if __name__ == "__main__":
    inference()