from abc import ABC
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from datasets.base import BaseDataset, collate_fn
from datasets import datasets as datasets_module
from datasets import *
from tqdm import tqdm  
from models import models as models_module
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
## each subset mapping
    "unrelated-image-color": "UnrelatedImageDataset",
    "unrelated-image-nature": "UnrelatedImageDataset",
    "unrelated-image-noise": "UnrelatedImageDataset",
    "instruction-enhancement-logic": "InstructionEnhanceLogicData",
    "adv-clean": "AdvUntarget",
    "adv-untarget": "AdvUntarget",
    "advglue-text": "AdvText",
    "advglue-related-image": "AdvText",
    "advglue-unrelated-image-color": "AdvText",
    "advglue-unrelated-image-nature": "AdvText",
    "advglue-unrelated-image-noise": "AdvText",
    "advglue-plus-text": "AdvText",
    "advglue-plus-related-image": "AdvText",
    "advglue-plus-unrelated-image-color": "AdvText",
    "advglue-plus-unrelated-image-nature": "AdvText",
    "advglue-plus-unrelated-image-noise": "AdvText",
    "dt-text": "OODText",
    "dt-related-image": "OODText",
    "dt-unrelated-image-color": "OODText",
    "dt-unrelated-image-nature": "OODText",
    "dt-unrelated-image-noise": "OODText",
    "confaide-text": "ConfAIde",
    "confaide-image": "ConfAIde",
    "confaide-unrelated-image-color": "ConfAIde",
    "confaide-unrelated-image-nature": "ConfAIde",
    "confaide-unrelated-image-noise": "ConfAIde",
    "stereo-classification-text": "StereoClassification",
    "stereo-classification-image": "StereoClassification",
    "stereo-classification-unrelated-image-color": "StereoClassification",
    "stereo-classification-unrelated-image-nature": "StereoClassification",
    "stereo-classification-unrelated-image-noise": "StereoClassification",
    "adv-target": "AdvTarget",
    "safebench": "SafeBenchDataset",
    "stereo-agreement-text": "StereoAgreement",
    "stereo-agreement-image": "StereoAgreement",
    "stereo-agreement-unrelated-image-color": "StereoAgreement",
    "stereo-agreement-unrelated-image-nature": "StereoAgreement",
    "stereo-agreement-unrelated-image-noise": "StereoAgreement",
    "g-text-assistance": "VisualMisleadingData",
    "g-text-misvisual": "VisualMisleadingData",
    "g-text-unrelated-image-color": "VisualMisleadingData",
    "g-text-unrelated-image-noise": "VisualMisleadingData",
    "g-text-unrelated-image-nature": "VisualMisleadingData",
    "g-text-none": "VisualMisleadingData",
    "g-text-misleading": "TextMisleadingData",
    "vizwiz-recognition": "VizWiz",
    "vizwiz-recognition-pri-query": "VizWiz",
    "vizwiz-drawline-recognition-pri-query": "VizWiz",
    "celebrities": "Celebrities",
    "crossmodal-jailbreak-text": "CrossModalJailbreakDataset",
    "crossmodal-jailbreak-unrelated": "CrossModalJailbreakDataset",
    "crossmodal-jailbreak-pos": "CrossModalJailbreakDataset",
    "crossmodal-jailbreak-neg": "CrossModalJailbreakDataset",
    "profession-pred": "ProfessionPred",
    "profession-pred-with-description": "ProfessionPred",
    "benchlmm-infrared": "OODSensor",
    "benchlmm-lxray": "OODSensor",
    "benchlmm-hxray": "OODSensor",
    "benchlmm-mri": "OODSensor",
    "benchlmm-ct": "OODSensor",
    "benchlmm-remote": "OODSensor",
    "benchlmm-driving": "OODSensor",
    "coco-o-cartoon": "OODArtistic",
    "coco-o-handmake": "OODArtistic",
    "coco-o-painting": "OODArtistic",
    "coco-o-sketch": "OODArtistic",
    "coco-o-tattoo": "OODArtistic",
    "coco-o-weather": "OODArtistic",
    "stereo-topic-classification-text": "StereoTopicClassification",
    "stereo-topic-classification-image": "StereoTopicClassification",
    "stereo-topic-classification-unrelated-image-color": "StereoTopicClassification",
    "stereo-topic-classification-unrelated-image-nature": "StereoTopicClassification",
    "stereo-topic-classification-unrelated-image-noise": "StereoTopicClassification",
    "enron-email-text": "EnronEmailDataset",
    "enron-email-image-oneinfo": "EnronEmailDataset",
    "enron-email-image-typeinfo": "EnronEmailDataset",
    "enron-email-image-allinfo": "EnronEmailDataset",
    "enron-email-unrelated-image-color": "EnronEmailDataset",
    "enron-email-unrelated-image-nature": "EnronEmailDataset",
    "enron-email-unrelated-image-noise": "EnronEmailDataset",
    "stereo-generation": "StereoGeneration",
    "subjective-preference-plain-text": "SubPreference",
    "subjective-preference-plain-image": "SubPreference",
    "subjective-preference-plain-unrelated-image-color": "SubPreference",
    "subjective-preference-plain-unrelated-image-nature": "SubPreference",
    "subjective-preference-plain-unrelated-image-noise": "SubPreference",
    "subjective-preference-force-text": "SubPreference",
    "subjective-preference-force-image": "SubPreference",
    "subjective-preference-force-unrelated-image-color": "SubPreference",
    "subjective-preference-force-unrelated-image-nature": "SubPreference",
    "subjective-preference-force-unrelated-image-noise": "SubPreference",
    "typographic-prompt-and-behavior": "TypographicDataset",
    "typographic-prompt": "TypographicDataset",
    "typographic-behavior": "TypographicDataset",
    "opimized-jailbreak-graphic": "TypographicDataset",
    "mm-safety-bench": "MMSafetyBenchDataset",
    "vision-preference": "VisionPreference",
    "visual-assistance-text": "VisualAssistance",
    "visual-assistance-image": "VisualAssistance",
    "visual-assistance-unrelated-image-color": "VisualAssistance",
    "visual-assistance-unrelated-image-nature": "VisualAssistance",
    "visual-assistance-unrelated-image-noise": "VisualAssistance",
    "instruction-enhancement-factual": "InstructionEnhanceFactualData",
    "d-mis-visual-confusion": "VisualConfusionData",
    "advanced-spatial": "AdvancedData",
    "advanced-temporal": "AdvancedData",
    "advanced-compare": "AdvancedData",
    "advanced-daily": "AdvancedData",
    "advanced-traffic": "AdvancedData",
    "advanced-causality": "AdvancedData",
    "advanced-math": "AdvancedData",
    "advanced-code": "AdvancedData",
    "advanced-translate": "AdvancedData",
    "nsfw-image-description": "NSFWDataset",
    "stereo-query-text": "StereoQuery",
    "stereo-query-image": "StereoQuery",
    "stereo-query-unrelated-image-color": "StereoQuery",
    "stereo-query-unrelated-image-nature": "StereoQuery",
    "stereo-query-unrelated-image-noise": "StereoQuery",
    "toxicity-prompt-text": "RealToxicityPromptsDataset",
    "toxicity-prompt-unrelated": "RealToxicityPromptsDataset",
    "toxicity-prompt-image": "RealToxicityPromptsDataset",
    "object-detection": "HodDataset",
    "risk-analysis": "HodDataset",
    "vispr-recognition": "Vispr",
    "vispr-leakage": "Vispr",
    "vispr-leakage-protected": "Vispr",
    "vispr-recognition-pri-query": "Vispr",
    "d-basic-object": "BasicData",
    "d-basic-attribute": "BasicData",
    "d-basic-scene": "BasicData",
    "g-basic-grounding": "BasicData",
    "g-basic-ocr": "BasicData"
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

    def get_dataset(self,follow_rules: bool,long_pipe: bool, danger: str) -> BaseDataset:
        dataset_cls = globals()[cls_mapping[self.dataset_id]]
        dataset = dataset_cls(self.dataset_id, follow_rules, long_pipe, self.safety_rules, danger)
        return dataset

    def get_dataloader(self,dataset) -> DataLoader:
        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)
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
            self, dataloader: DataLoader, long_pipe: bool, log_file: str,  **generate_kwargs
        ) -> List[Dict[str, Any]]:
            print("len(self.dataset): ", len(dataloader.dataset))
            responses = []

            log_dir = os.path.dirname(log_file)
            file_name = os.path.basename(log_file) 

            if long_pipe:
                print("-----------------------------")
                print("Start two stage answering...")
                print("-----------------------------")
                print("Start generating relation...") if self.follow_rules else print("Start generating final answer...")
            else:
                print("-------------------------------------")
                print("Start directly reason with rules...")
                print("-------------------------------------")
            # question_id = 0
            for batch_data in tqdm(dataloader, desc="Processing Batches", unit="batch"):
                for data in tqdm(batch_data, desc="Processing Data", unit="data", leave=False):
                    message = data["message"]                    
                    response = self.follow_model.chat(messages=message, **generate_kwargs) if self.follow_rules else self.reason_model.chat(messages=message, **generate_kwargs)
                    output = {
                    "content": message[0]["content"],
                    "response": response.content,
                }
                    responses.append(output)

            if long_pipe:
                relation_path = os.path.join(log_dir, "relation")
                final_answer_path = os.path.join(log_dir, "final_answer")
                os.makedirs(relation_path, exist_ok=True)
                os.makedirs(final_answer_path, exist_ok=True)
                relation_file = os.path.join(relation_path, file_name)
                answer_file = os.path.join(final_answer_path, file_name)
                with open(relation_file if self.follow_rules else answer_file, "w", encoding="utf-8") as rf:
                    json.dump(responses, rf, ensure_ascii=False, indent=4)
            else:
                rules_reason_path = os.path.join(log_dir, "rules_reason")
                os.makedirs(rules_reason_path, exist_ok=True)
                rules_reason_file = os.path.join(rules_reason_path, file_name)
                with open(rules_reason_file, "w", encoding="utf-8") as rf:
                    json.dump(responses, rf, ensure_ascii=False, indent=4)                

            return responses

    def pipeline(self) -> None:
        self.follow_dataset = self.get_dataset(follow_rules=True, long_pipe=True, danger=None)
        follow_dataloader = self.get_dataloader(self.follow_dataset)
        self.follow_model = self.get_model(follow_rules=True)
        print("Start safety-aware rationale generating....")
        danger = self.generate(follow_dataloader,self.log_file, **self.generation_kwargs)
        print("Safety-aware rationale generation end, start reasoning...")
        self.follow_rules = False
        self.reason_dataset = self.get_dataset(follow_rules=False, long_pipe=True, danger=danger)
        self.reason_model = self.get_model(follow_rules=False)
        reason_dataloader = self.get_dataloader(self.reason_dataset)
        final_responses = self.generate(reason_dataloader,long_pipe=True,log_file=self.log_file, **self.generation_kwargs)
        print("Reasoning end, start evaluating...")
        
        # self.evaluators = self.get_evaluators()
        # results = self.eval(final_responses)
        # self.save_results(final_responses)
    
    def pipeline_ref(self) -> None:
        self.follow_rules = False
        self.reason_dataset = self.get_dataset(follow_rules=False, long_pipe=False, danger=None)
        reason_dataloader = self.get_dataloader(self.reason_dataset)
        self.reason_model = self.get_model(follow_rules=False)
        print("Start to generate the response directly without reasoning....")
        final_responses=self.generate(reason_dataloader,long_pipe=False,log_file=self.log_file, **self.generation_kwargs)
        print("Reasoning end, start evaluating...")
        # self.evaluators = self.get_evaluators()
        # results = self.eval(final_responses)
        # self.save_results(final_responses)