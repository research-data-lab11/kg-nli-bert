# -*- coding: utf-8 -*-
import csv
import re

def clean_concept(concept):
    parts = concept.split('/')
    meaningful_parts = [p for p in parts if not re.fullmatch(r'(n|v|a|r|wn|wikt|wp|en_\d+)', p)]
    cleaned = ' '.join(meaningful_parts).replace('_', ' ')
    return cleaned.strip()

input_file = "/path/to/your/workspace/conceptnet-assertions-5.7.0.csv/assertions.csv"
output_file = "/path/to/your/workspace/ConceptNet_to_spo.spo"

seen_triples = set()  

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    reader = csv.reader(infile, delimiter='\t')
    count = 0

    for row in reader:
        if len(row) < 4:
            continue
        relation_uri = row[1]
        subject_uri = row[2]
        object_uri = row[3]

        if subject_uri.startswith("/c/en/") and object_uri.startswith("/c/en/"):
            subject_raw = subject_uri.replace("/c/en/", "")
            object_raw = object_uri.replace("/c/en/", "")

            subject = clean_concept(subject_raw)
            object_ = clean_concept(object_raw)
            predicate = relation_uri.split("/")[-1]

            triple = (subject, predicate, object_)

            if triple not in seen_triples: 
                outfile.write(f"{subject}\t{predicate}\t{object_}\n")
                seen_triples.add(triple)
                count += 1


