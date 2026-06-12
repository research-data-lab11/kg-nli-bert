# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:35:35 2025

@author: AILAB
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List
import pandas as pd

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import double_sentence_featurize, labels_to_bimap
from jiant.utils.python.io import read_json

@dataclass
class Example(BaseExample):
    guid: str
    input_premise: str
    input_hypothesis: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_premise=tokenizer.tokenize(self.input_premise),
            input_hypothesis=tokenizer.tokenize(self.input_hypothesis),
            label_id=SickEnrichedTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_premise: List
    input_hypothesis: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.input_premise,
            input_tokens_b=self.input_hypothesis,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )

@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list

@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class SickEnrichedTask(Task):
  
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch
    
    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples((self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples((self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
       examples = []
    
    # Έλεγχος αν τα δεδομένα είναι σε μορφή JSON ή txt
       if isinstance(lines, str) and lines.endswith(".txt"):  # Αν το 'lines' είναι διαδρομή σε αρχείο txt
          with open(lines, 'r') as file:
            lines = file.readlines()
    
       for line in lines:
        # Αν το line είναι μια γραμμή κειμένου
         if isinstance(line, str):
            # Διαχωρισμός της γραμμής (π.χ., αν το txt αρχείο έχει την μορφή "sentence1 \t sentence2 \t label")
            parts = line.strip().split("\t")
            if len(parts) == 3:
                sentence1, sentence2, label = parts
            else:
                raise ValueError(f"Μη αναμενόμενη γραμμή στο αρχείο: {line}")

            # Δημιουργία του παραδείγματος
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, len(examples)),
                    input_premise=sentence1,
                    input_hypothesis=sentence2,
                    label=label if set_type != "test" else cls.LABELS[-1],
                )
            )
        # Αν το line είναι ήδη μορφή λεξικού (π.χ. από JSON)
         elif isinstance(line, dict):
            examples.append(
                Example(
                    guid="%s-%s" % (set_type, len(examples)),
                    input_premise=line["sentence1"],
                    input_hypothesis=line["sentence2"],
                    label=line["label"] if set_type != "test" else cls.LABELS[-1],
                )
            )

       return examples
