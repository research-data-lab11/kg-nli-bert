# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

def load_jsonl(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset

def save_jsonl(path, dataset):
    with open(path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def filter_dataset(dataset, max_len=128):
    filtered = []
    removed = 0
    for item in dataset:
        too_long = False
        for field in ["premise", "hypothesis"]:
            tokens = tokenizer.encode(item[field], add_special_tokens=True)
            if len(tokens) > max_len:
                too_long = True
                break
        if not too_long:
            filtered.append(item)
        else:
            removed += 1
    return filtered

train_path = "/path/to/your/workspace/name_workspace/tasks/data/scitail/train.jsonl"
save_path = "/path/to/your/workspace/name_workspace/tasks/data/scitail/train_filtered.jsonl"

dataset = load_jsonl(train_path)
filtered_dataset = filter_dataset(dataset, max_len=128)
save_jsonl(save_path, filtered_dataset)
