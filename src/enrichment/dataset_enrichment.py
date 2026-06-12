#!/usr/bin/env python3
# -*- coding: utf-8 -*-

EXP_DIR = '/path/to/your/workspace/name_workspace'

import json
import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Διαθέσιμες GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

def get_embedding(text):
    with torch.no_grad():
        embedding = model.module.encode(text, convert_to_tensor=True)
        torch.cuda.empty_cache()  
    return embedding.cpu().detach().numpy()


def load_graph(file_path):
    G = nx.DiGraph()  
    edges = []  
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                subject, predicate, obj = parts
                edges.append((subject, obj, {'relation': predicate})) 
                pass
    G.add_edges_from(edges)  
    return G


graph = load_graph(f"{EXP_DIR}/kg/WordNet.spo")

import gc       
def enrich_sentence_by_token(sentence, graph, similarity_threshold=0.70):
    sentence_embedding = get_embedding(sentence).reshape(1, -1)
    enriched_sentence = sentence
    tokens = sentence.split()

    subjects = set([triple[0] for triple in graph.edges])
    ngrams = [' '.join(tokens[i:j]) for i in range(len(tokens)) for j in range(i+1, len(tokens)+1)]
    for ngram in ngrams:
        if ngram.lower() in subjects:
            candidate_triples = []
            for neighbor in graph.neighbors(ngram.lower()):
                relation = graph[ngram.lower()][neighbor]['relation']
                if graph[ngram.lower()][neighbor]:
                    object = neighbor 
                    triple_str = f"({ngram} {relation} {object})"
                    triple_embedding = get_embedding(triple_str).reshape(1, -1)
                    similarity = cosine_similarity(sentence_embedding, triple_embedding)[0][0]
                    candidate_triples.append((similarity, triple_str))
                    del triple_embedding  
                    torch.cuda.empty_cache() 
            candidate_triples.sort(reverse=True, key=lambda x: x[0])
            candidate_triples = [triple for triple in candidate_triples if triple[0] >= similarity_threshold]
            print(f"N-gram: {ngram} - Similarity threshold: {similarity_threshold}")
            for i, (similarity, triple) in enumerate(candidate_triples[:2]):
                print(f"  Top {i+1} Triple: {triple} | Similarity: {similarity:.4f}")
            for _, triple in candidate_triples[:2]:
                enriched_sentence += f" {triple}"
    del sentence_embedding  
    gc.collect()  
    return enriched_sentence

def load_local_dataset_jsonl(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))       
    return dataset

#def load_local_dataset_txt(filepath):
    #dataset = []
    #with open(filepath, 'r', encoding='utf-8') as file:
        #for line in file:
            #parts = line.strip().split('\t')
            #if len(parts) == 3:
                #dataset.append({'sentence1': parts[0], 'sentence2': parts[1], 'label': parts[2]})
    #return dataset
    
train_dataset = load_local_dataset_jsonl(f"{EXP_DIR}/tasks/data/snli/train.jsonl")
#train_dataset = load_local_dataset_txt(f"{EXP_DIR}/tasks/data/dnli/train.txt")

# Εμπλουτισμός του dataset
def enrich_dataset_jsonl(dataset, graph, similarity_threshold=0.7):
    enriched_dataset = []
    for i, item in enumerate(dataset):
        enriched_sentence1 = enrich_sentence_by_token(item["premise"], graph, similarity_threshold)
        enriched_sentence2 = enrich_sentence_by_token(item["hypothesis"], graph, similarity_threshold)
        enriched_dataset.append({"premise": enriched_sentence1, "hypothesis": enriched_sentence2, "label": item["label"]})
        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return enriched_dataset

#def enrich_dataset_txt(dataset, graph, similarity_threshold=0.7):
   # enriched_dataset = []
    #for item in dataset:
        #enriched_sentence1 = enrich_sentence_by_token(item["sentence1"], graph, similarity_threshold)
        #enriched_sentence2 = enrich_sentence_by_token(item["sentence2"], graph, similarity_threshold)
        #enriched_dataset.append({"sentence1": enriched_sentence1, "sentence2": enriched_sentence2, "label": item["label"]})
    #return enriched_dataset


enriched_train_data = enrich_dataset_jsonl(train_dataset, graph, similarity_threshold=0.7)
#enriched_train_data = enrich_dataset_txt(train_dataset, graph, similarity_threshold=0.7)

train_output_path = f"{EXP_DIR}/tasks/data/snli/enriched_train_wordnet.json"
with open(train_output_path, "w", encoding="utf-8") as f:
    json.dump(enriched_train_data, f)
