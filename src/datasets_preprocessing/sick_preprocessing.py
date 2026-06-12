# -*- coding: utf-8 -*-

import csv
import os


input_file = '/path/to/your/workspace/sick.txt'

# Καθορισμός του φακέλου εξόδου
output_dir = '/path/to/your/workspace/sick'  
os.makedirs(output_dir, exist_ok=True)


output_files = {
    'TRAIN': os.path.join(output_dir, 'train.txt'),
    'TRIAL': os.path.join(output_dir, 'dev.txt'),
    'TEST': os.path.join(output_dir, 'test.txt')
}


with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter='\t') 
    
 
    output_fps = {
        key: open(output_files[key], 'w', encoding='utf-8') for key in output_files
    }
    
    for row in reader:
        sentence_A = row['sentence_A']
        sentence_B = row['sentence_B']
        entailment_label = row['entailment_label']
        semeval_set = row['SemEval_set'].upper()  
        if semeval_set in output_fps:
            fp = output_fps[semeval_set]
            fp.write(f"{sentence_A}\t{sentence_B}\t{entailment_label}\n")
    for fp in output_fps.values():
        fp.close()



