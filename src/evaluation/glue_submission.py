# -*- coding: utf-8 -*-

import sys
import numpy as np
import numpy.core
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.multiarray"] = np.core.multiarray
sys.modules["numpy._core.numeric"] = np.core.numeric
import os
import torch
import pandas as pd

folder = r"/path/to/your/workspace/name_workspace/runs/qnli_enriched2_6"
input_file = "test_preds.p"

input_path = os.path.join(folder, input_file)
output_path = os.path.join(folder, "QNLI.tsv")
data = torch.load(input_path, map_location="cpu", weights_only=False)
task_data = data["qnli_enriched2"]
preds = task_data["preds"]
guids = task_data["guids"]
label_map = {
    0: "entailment",
    1: "not_entailment"
}
indices = [int(str(guid).split("-")[-1]) for guid in guids]
pred_labels = [label_map[int(pred)] for pred in preds]

df = pd.DataFrame({
    "index": indices,
    "prediction": pred_labels
})

df = df.sort_values("index")
df.to_csv(output_path, sep="\t", index=False)

# For QNLI  0: "entailment", 1: "not_entailment"
# For RTE  0: "entailment", 1: "not_entailment"