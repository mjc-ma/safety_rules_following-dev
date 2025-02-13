# dev: PYTHONPATH=. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=3000 python extract_mapping.py

import os
import json
import ast

DATASETS_DIR = "/mnt/petrelfs/majiachen/project/safety_rules_following-dev/datasets"
OUTPUT_FILE = os.path.join(DATASETS_DIR, "dataset_mapping.json")

def extract_dataset_mapping(directory):
    dataset_mapping = {}

    for filename in os.listdir(directory):
        if filename.endswith(".py"):  
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=file_path)
                except SyntaxError as e:
                    print(f"Syntax error in {filename}: {e}")
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):  
                        class_name = node.name
                        dataset_ids = None

                        for body_item in node.body:
                            # breakpoint()
                            if isinstance(body_item, ast.AnnAssign):  # 处理类型注解赋值
                                if isinstance(body_item.target, ast.Name) and body_item.target.id == "dataset_ids":
                                    if isinstance(body_item.value, ast.List):
                                        dataset_ids = [elt.s for elt in body_item.value.elts if isinstance(elt, ast.Str)]
                            elif isinstance(body_item, ast.Assign):  # 处理普通赋值
                                for target in body_item.targets:
                                    if isinstance(target, ast.Name) and target.id == "dataset_ids":
                                        if isinstance(body_item.value, ast.List):
                                            dataset_ids = [elt.s for elt in body_item.value.elts if isinstance(elt, ast.Str)]

                        if dataset_ids:
                            for dataset_id in dataset_ids:
                                dataset_mapping[dataset_id] = class_name

    return dataset_mapping

if __name__ == "__main__":
    mapping = extract_dataset_mapping(DATASETS_DIR)

    # 保存为 JSON 文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as json_file:
        json.dump(mapping, json_file, indent=4, ensure_ascii=False)

    print(f"Dataset mapping saved to {OUTPUT_FILE}")
