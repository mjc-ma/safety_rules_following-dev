from abc import ABC
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from datasets.base import BaseDataset, collate_fn
from datasets import datasets as datasets_module
from models import models as models_module
from models import models_for_Qwen as models_one
from models.base import BaseChat
# from evaluators.base import SequentialEvaluator
import numpy as np
import warnings
import json
from typing import List, Dict, Any
import os
cls_mapping = {
    "mm-safety-bench": "MMSafetyBenchDataset",
    "llava-1.5-7b-hf": "LLaVAHFChat",
    "llama-3-2-chat":"LlamaChat",
    "Qwen2.5-VL-7B-Instruct":"Qwen2Chat",
    "gpt-4o":"OpenAIChat"
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
            model_cls = models_one.__dict__[cls_mapping[self.follow_model_id]]
            model = model_cls(self.follow_model_id)
        else:
            model_cls = models_module.__dict__[cls_mapping[self.reason_model_id]]
            model = model_cls(self.reason_model_id)
        return model

    def get_dataset(self,follow_rules: bool, long_pipe: bool, danger: str) -> BaseDataset:
        dataset_cls = datasets_module.__dict__[cls_mapping[self.dataset_id]]
        # breakpoint()
        if follow_rules == True:
            dataset = dataset_cls(
                self.dataset_id, follow_rules, long_pipe, self.safety_rules, danger
        )
            # breakpoint()
        else:
            dataset = dataset_cls(
                self.dataset_id, follow_rules, long_pipe, self.safety_rules, danger
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

    # def generate_old(
    #     self, dataloader: DataLoader, **generate_kwargs
    # ) -> List[Dict[str, Any]]:
    #     print("len(self.dataset): ", len(dataloader.dataset))
    #     responses = []
    #     for batch_data in dataloader:
    #         for data in batch_data:
    #             """
    #             # for multimodal data
    #             message = [
    #                 {
    #                     "role": "user",
    #                     "content": {
    #                         "image_path": ...,
    #                         "text": ...
    #                     }
    #                 }
    #             ]
    #             """
    #             message = data["message"]
    #             flag = 1
    #             if self.follow_rules == True:
    #                 response = self.follow_model.chat(messages=message, **generate_kwargs)
    #             else :
    #                 response = self.reason_model.chat(messages=message, **generate_kwargs)
    #                 flag = 0

    #             output = {
    #                 "content": message[0]["content"],
    #                 "response": response.content,
    #             }
    #             # breakpoint()
    #             #print("output:", output)
    #             responses.append(output)
    #     return responses
  
    def generate(
            self, dataloader: DataLoader,  **generate_kwargs
        ) -> List[Dict[str, Any]]:
            print("len(self.dataset): ", len(dataloader.dataset))
            responses = []
            #修改下面的变量
            json_path = "/mnt/petrelfs/fanyuyu/safety_rules_following-dev/data/processed_questions_qwen_long/03-Malware_Generation.json"
            # 加载现有JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            question_id = 0

            for batch_data in dataloader:
                for data in batch_data:
                    message = data["message"]
                    flag = 1
                    
                    # 获取当前问题的ID
                    #question_id = str(data["question_id"])  # 假设data包含question_id字段
                    print(self.follow_rules)
                    if self.follow_rules:
                        print("Yessss")
                        response = self.follow_model.chat(messages=message, **generate_kwargs)
                    else:
                        print("\n")
                        print("hello")
                        print("\n")
                        response = self.reason_model.chat(messages=message, **generate_kwargs)
                        flag = 0

                    output = {
                        "content": message[0]["content"],
                        "response": response.content,
                    }
                    if flag == 1:
                        # 更新JSON数据
                        question_id = str(question_id)
                        print("aaaaaaaaaaaaaaaaaa")
                        print(question_id)
                        print("////////////////")
                        if question_id in json_data:
                            # 确保ans结构存在
                            if "rationale" not in json_data[question_id]:
                                json_data[question_id]["rationale"] = {}
                            if "Qwen2.5_llama3.2" not in json_data[question_id]["rationale"]:
                                json_data[question_id]["rationale"]["Qwen2.5_llama3.2"] = {"reason": ""}
                                
                            # 替换响应内容
                            json_data[question_id]["rationale"]["Qwen2.5_llama3.2"]["reason"] = response.content

                    if flag == 0:
                        # 更新JSON数据
                        question_id = str(question_id)
                        print("aaaaaaaaaaaaaaaaaa")
                        print(question_id)
                        print("////////////////")
                        if question_id in json_data:
                            # 确保ans结构存在
                            if "ans" not in json_data[question_id]:
                                json_data[question_id]["ans"] = {}
                            if "Qwen2.5_llama3.2" not in json_data[question_id]["ans"]:
                                json_data[question_id]["ans"]["Qwen2.5_llama3.2"] = {"text": ""}
                                
                            # 替换响应内容
                            json_data[question_id]["ans"]["Qwen2.5_llama3.2"]["text"] = response.content

                    question_id = int(question_id)
                    question_id = question_id + 1

                    responses.append(output)

            # 保存更新后的JSON数据
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            return responses

    def pipeline(self) -> None:
        self.follow_dataset = self.get_dataset(follow_rules=True,long_pipe = True, danger=None)
        follow_dataloader = self.get_dataloader(self.follow_dataset)
        self.follow_model = self.get_model(follow_rules=True)
        print("Start safety-aware rationale generating....")
        danger = self.generate(follow_dataloader, **self.generation_kwargs)
        #breakpoint()
        print("Safety-aware rationale generation end, start reasoning...")
        self.follow_rules = False
        self.reason_dataset = self.get_dataset(follow_rules=False,long_pipe = True,danger=danger)
        self.reason_model = self.get_model(follow_rules=False)
        reason_dataloader = self.get_dataloader(self.reason_dataset)
        final_responses = self.generate(reason_dataloader, **self.generation_kwargs)
        print("Reasoning end, start evaluating...")
        
        self.evaluators = self.get_evaluators()
        results = self.eval(final_responses)
        self.save_results(final_responses)
    
    def pipeline_ref(self) -> None:
        self.follow_rules = False
        self.reason_dataset = self.get_dataset(follow_rules=True,long_pipe = False,danger = None)
        reason_dataloader = self.get_dataloader(self.reason_dataset)
        #short_pipe by follow_model(follow_rules=True) or reason_model(follow_rules=False)?
        self.reason_model = self.get_model(follow_rules=True)
        print("Start to generate the response directly without reasoning....")
        final_responses=self.generate(reason_dataloader, **self.generation_kwargs)
        print("Reasoning end, start evaluating...")
        self.evaluators = self.get_evaluators()
        results = self.eval(final_responses)
        self.save_results(final_responses)