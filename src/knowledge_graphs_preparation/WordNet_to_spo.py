# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn

def extract_all_triplets():
    triplets = set() 
    unique_lemmas = set()
    unique_synsets = set()

 
    for synset in wn.all_synsets():
        unique_synsets.add(synset)

    
        for lemma in synset.lemmas():
            subject = lemma.name() 
            unique_lemmas.add(subject)

            for synonym in synset.lemmas():
                if synonym != lemma:  
                    triplets.add((subject, "synonym", synonym.name()))

           
            for lemma_antonym in synset.lemmas():
                if lemma_antonym.antonyms():
                    for antonym in lemma_antonym.antonyms():
                        triplets.add((subject, "antonym", antonym.name()))

           
            for hypernym in synset.hypernyms():
                triplets.add((subject, "has_hypernym", hypernym.name()))

         
            for hyponym in synset.hyponyms():
                triplets.add((subject, "has_hyponym", hyponym.name()))

         
            for meronym in synset.part_meronyms():
                triplets.add((subject, "has_part", meronym.name()))
            for meronym in synset.member_meronyms():
                triplets.add((subject, "has_member", meronym.name()))
            for meronym in synset.substance_meronyms():
                triplets.add((subject, "has_substance", meronym.name()))

           
            for holonym in synset.part_holonyms():
                triplets.add((subject, "part_of", holonym.name()))
            for holonym in synset.member_holonyms():
                triplets.add((subject, "member_of", holonym.name()))
            for holonym in synset.substance_holonyms():
                triplets.add((subject, "substance_of", holonym.name()))

          
            for entailment in synset.entailments():
                triplets.add((subject, "entails", entailment.name()))

           
            if synset.pos() == 'v' and hasattr(synset,'troponyms'): 
               for troponym in synset.troponyms():
                  triplets.add((subject, "is_a", troponym.name()))

        
            for attribute in synset.attributes():
                triplets.add((subject, "has_attribute", attribute.name()))

         
            for similar in synset.similar_tos():
                triplets.add((subject, "similar_to", similar.name()))


           
            for entailed in synset.entailments():
                triplets.add((entailed.name(), "entailed_by",subject))

            

    return triplets, unique_lemmas, unique_synsets


def save_as_spo(triplets, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(f"{triplet[0]}\t{triplet[1]}\t{triplet[2]}\n")


triplets, unique_lemmas, unique_synsets = extract_all_triplets()


output_path = 'WordNet_to_spo.spo'
save_as_spo(triplets, output_path)


