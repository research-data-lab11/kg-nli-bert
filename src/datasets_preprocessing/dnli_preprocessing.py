# -*- coding: utf-8 -*-

import json
import os

input_paths = {
    'train': '/path/to/your/workspace/dnli/dialogue_nli_train.jsonl',
    'dev': '/path/to/your/workspace/dnli/dialogue_nli_dev.jsonl',
    'test': '/path/to/your/workspace/dnli/dialogue_nli_test.jsonl'
}

output_dir = '/path/to/your/workspace/dnli'  
os.makedirs(output_dir, exist_ok=True)


files = [
    ('train', 'train.txt'),
    ('dev', 'dev.txt'),
    ('test', 'test.txt')
]


for input_key, output_file in files:
    input_file = input_paths[input_key]  
    with open(input_file, 'r') as infile, open(os.path.join(output_dir, output_file), 'w') as outfile:
        for line in infile:
            data = json.loads(line)  
            
           
            label = data.get('label', '')
            sentence1 = data.get('sentence1', '')
            sentence2 = data.get('sentence2', '')
            
            
            outfile.write(f"{sentence1} {sentence2} {label} \n")