from abc import ABC
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from datasets.base import BaseDataset, collate_fn
from datasets import datasets as datasets_module
from models import models as models_module
from models.base import BaseChat
# from evaluators.base import SequentialEvaluator
import numpy as np
import warnings
import json
import os
cls_mapping = {
    "mm-safety-bench": "MMSafetyBenchDataset",
    "llama-3-2-chat": "LlamaChat",
}
batch_size = 1

class BaseTask(ABC):
    def __init__(
        self,
        dataset_id: str,
        follow_model_id: str,
        reason_model_id: str,
        follow_rules: bool,
        safety_rules: str,
        dataset_categories: str,
        generation_kwargs: Optional[Dict] = {},
        evaluator_seq_cfgs: List = [],
        log_file: Optional[str] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.follow_model_id = follow_model_id
        self.reason_model_id = reason_model_id
        self.follow_rules = follow_rules
        self.safety_rules = safety_rules
        self.dataset_categories = dataset_categories
        self.evaluator_seq_cfgs = evaluator_seq_cfgs
        self.generation_kwargs = generation_kwargs
        self.log_file = log_file

    def get_model(self,follow_rules: bool) -> BaseChat:
        if follow_rules == True:
            model_cls = models_module.__dict__[cls_mapping[self.follow_model_id]]
            model = model_cls(self.follow_model_id)
        else:
            model_cls = models_module.__dict__[cls_mapping[self.reason_model_id]]
            model = model_cls(self.reason_model_id)
        return model

    def get_dataset(self,follow_rules: bool, danger: str) -> BaseDataset:
        dataset_cls = datasets_module.__dict__[cls_mapping[self.dataset_id]]
        # breakpoint()
        if follow_rules == True:
            dataset = dataset_cls(
                self.dataset_id, follow_rules, self.safety_rules, danger
        )
            # breakpoint()
        else:
            dataset = dataset_cls(
                self.dataset_id, follow_rules, self.safety_rules, danger
        )
        # breakpoint()
        return dataset

    def get_dataloader(self,dataset) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        return dataloader
   
    def save_results(self, results: Dict[str, Any]) -> None:
        scatter_keys = [
            key for key, value in results.items() if isinstance(value, Sequence)
        ]
        summary_keys = [
            key
            for key, value in results.items()
            if not isinstance(value, Sequence) and value is not None
        ]

        if scatter_keys:
            seq_len = len(results[scatter_keys[0]])
            for key in scatter_keys:
                assert len(results[key]) == seq_len

        per_sample_results = []
        for idx in range(seq_len):
            per_sample_result = {}
            for key in scatter_keys:
                per_sample_result[key] = results[key][idx]
            per_sample_results.append(per_sample_result)

        formatted_results = {}
        formatted_results["total_results"] = {}
        formatted_results["per_sample_results"] = per_sample_results
        for key in summary_keys:
            formatted_results["total_results"][key] = (
                float(results[key])
                if isinstance(results[key], (np.floating, np.integer))
                else results[key]
            )

        if self.log_file is not None:
            # check if the folder exists
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))

            with open(self.log_file, "w") as f:
                json.dump(formatted_results, f, indent=4)

    def generate(
        self, dataloader: DataLoader, **generate_kwargs
    ) -> List[Dict[str, Any]]:
        print("len(self.dataset): ", len(dataloader.dataset))
        responses = []
        for batch_data in dataloader:
            for data in batch_data:
                """
                # for multimodal data
                message = [
                    {
                        "role": "user",
                        "content": {
                            "image_path": ...,
                            "text": ...
                        }
                    }
                ]
                """
                message = data["message"]
                if self.follow_rules == True:
                    response = self.follow_model.chat(messages=message, **generate_kwargs)
                else :
                    response = self.reason_model.chat(messages=message, **generate_kwargs)

                output = {
                    "content": message[0]["content"],
                    "response": response.content,
                }
                # breakpoint()
                print("output:", output)
                responses.append(output)
        return responses

    def pipeline(self) -> None:
        self.follow_dataset = self.get_dataset(follow_rules=True,danger=None)
        follow_dataloader = self.get_dataloader(self.follow_dataset)
        self.follow_model = self.get_model(follow_rules=True)
        print("Start safety-aware rationale generating....")
        danger = self.generate(follow_dataloader, **self.generation_kwargs)
        breakpoint()
        print("Safety-aware rationale generation end, start reasoning...")
        self.follow_rules == False
        self.reason_dataset = self.get_dataset(follow_rules=False,danger=danger)
        self.reason_model = self.get_model(follow_rules=False)
        reason_dataloader = self.get_dataloader(self.reason_dataset)
        final_responses = self.generate(reason_dataloader, **self.generation_kwargs)
        print("Reasoning end, start evaluating...")
        
        self.evaluators = self.get_evaluators()
        results = self.eval(final_responses)
        self.save_results(final_responses)
