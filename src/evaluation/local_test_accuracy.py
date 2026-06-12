#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import pandas as pd
import os
from sklearn.metrics import accuracy_score 

import numpy as np

df = pd.read_csv('/path/to/your/workspace/name_workspace/tasks/data/dnli_enriched/test.txt', sep="\t", header=None)
#df = pd.read_json('/path/to/your/workspace/name_workspace/tasks/data/snli_enriched2/test.jsonl',lines=True)

s = pd.Series(df.iloc[:, 2])  
s = s.replace({"positive":0,"negative":1,"neutral":2})
df.iloc[:, 2] = s  

#s = pd.Series(df['label'])
#s = s.replace({"neutral": 2, "entailment": 1,"contradiction": 0})
#df['label'] = s

#df = df[df['label'] != -1] #snli

path = os.path.join("/path/to/your/workspace/name_workspace/runs/dnli_enriched19")
test_path = os.path.join(path, 'test_preds.p')
test_preds = torch.load(test_path, weights_only=False)
predictions = test_preds['dnli_enriched']['preds']
num_guids = len(test_preds['dnli_enriched']['guids'])
num_preds = len(test_preds['dnli_enriched']['preds'])
s= np.array(s, dtype=int)
predictions = np.array(predictions, dtype=int)

accuracy = accuracy_score(s, predictions)
#accuracy = accuracy_score(df['label'], predictions)

accuracy_file_path = "/path/to/your/workspace/name_workspace/runs/dnli_enriched19/test_accuracy.txt"

with open(accuracy_file_path, 'w') as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")

# For  SciTail neutral:1, entails:0
# For SNLI entailment:1, contradiction:0, neutral:2
# For DNLI  positive:0, negative:1, neutral:2
# For SICK ENTAILMENT:2, CONTRADICTION:0, NEUTRAL:1



