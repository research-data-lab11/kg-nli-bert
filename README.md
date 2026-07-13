# KG-NLI-BERT

Understanding and inferring logical relations from natural language is a fundamental human ability, yet it remains a challenging problem for computational systems. Natural Language Inference (NLI) requires not only the interpretation of words and syntactic structures, but also the use of broader semantic and world knowledge, which modern language models do not always capture effectively.

This repository accompanies the paper on knowledge graph-based enrichment for NLI. The implementation is based on the `jiant` framework, which was used for managing NLI tasks, fine-tuning pretrained language models, and evaluating the experimental configurations.

This paper investigates whether the integration of external knowledge, represented as triplets from Knowledge Graphs (KGs), can improve the performance of a pretrained language model on NLI tasks. Six benchmark datasets, namely **RTE**, **QNLI**, **SICK**, **SNLI**, **DNLI**, and **SciTail**, were enriched with information extracted from **AristoTuple** and **WordNet**, while **ConceptNet** was additionally used for the SNLI and SciTail datasets. The enrichment was applied only to the training data, through a strict relevance-based selection mechanism using `paraphrase-MiniLM-L6-v2` Sentence Transformer embeddings and cosine similarity. The resulting data were used to fine-tune the `BERT-base-uncased` model.

The experimental results show that the contribution of external knowledge varies across datasets, leading to a range of performance changes rather than a uniform improvement. This variation appears to depend not only on the source and relevance of the integrated knowledge, but also on dataset-specific characteristics, such as sentence length, content, linguistic structure, and the way each dataset was constructed. In particular, the SICK dataset showed more limited benefits from knowledge enrichment, suggesting that certain datasets may be less compatible with this type of external information.

## System Implementation

### Environment Setup

The system was implemented using the `jiant` framework, a flexible toolkit for experimenting with language models across multiple NLP tasks. In this work, `jiant` was used to support the training and evaluation of BERT-based model on both original and knowledge-enriched NLI datasets.

The experimental environment was configured on a Windows 10 machine using Windows Subsystem for Linux (WSL), in order to provide a stable Linux-based setup for installing and running `jiant` and its dependencies. The experiments were executed on a machine equipped with two GPUs, an NVIDIA GeForce RTX 3080 Ti and an NVIDIA GeForce RTX 2080 Ti, which enabled efficient training and evaluation of the models.

Python 3.11 was used in a virtual environment to isolate the project's dependencies and ensure compatibility between required libraries. Since the original `jiant` dependency file had conflicts between library versions, we modified the `requirements-no-torch.txt` file.

The following steps describe the complete setup process used for preparing the project environment through WSL.

```bash
# Move to the directory where you want to set up the project
cd /path/to/your/workspace

# Create a working directory
mkdir name_workspace
cd name_workspace

# Clone this repository
git clone https://github.com/research-data-lab11/kg-nli-bert.git

# Move into the cloned project repository
cd kg-nli-bert
```

The original `jiant` repository is then cloned inside the project repository. The `jiant` folder is not included in this repository, because it is obtained from its original source and then modified using the files provided here.

```bash
# Clone the original jiant repository inside the project directory
git clone https://github.com/nyu-mll/jiant.git
```

```bash
# Update the package list in the WSL/Linux environment
sudo apt update

# Install Python 3.11 and the required packages for creating a virtual environment
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create a Python 3.11 virtual environment for this project
python3.11 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate
```

Before installing the required libraries, replace the original `jiant/requirements-no-torch.txt` file with the modified version provided in this repository:

```bash
cp modified_jiant/requirements-no-torch.txt jiant/requirements-no-torch.txt
```

Then move into the `jiant` directory and install the modified dependencies:

```bash
cd jiant
pip install -r requirements-no-torch.txt
```

After installing the required dependencies, Spyder can optionally be installed and used as an IDE for editing and running the project scripts within the same WSL environment. This step is not required by `jiant`, but it was used during the implementation for code development and experiment execution.

```bash
# Install Spyder IDE dependencies
sudo apt install qt5-qmake qtbase5-dev

# Install Spyder inside the active virtual environment
pip install spyder

# Start Spyder
chmod 0700 /run/user/1000
spyder &
```

### Knowledge Graph Preparation

External knowledge was prepared in the form of subject–predicate–object triplets (`.spo` files), so that it could be directly used during the dataset enrichment process. The prepared knowledge graph files are stored in the `kg/` directory.

Three knowledge sources were used:

- **AristoTuple**
- **WordNet**
- **ConceptNet**

The preparation of the knowledge graphs followed two different procedures. For **AristoTuple** and **ConceptNet**, the original resources were obtained from their official sources as structured files. These files were then parsed and filtered using Python scripts in order to keep the relevant fields and convert them into subject–predicate–object (`.spo`) triplets.

In contrast, **WordNet** was not used as a pre-existing triplet file. Instead, a WordNet-based knowledge graph was generated programmatically through Python and the NLTK library. The script traverses WordNet synsets and extracts lexical-semantic relations, such as synonyms, antonyms, hypernyms, hyponyms, and meronym–holonym relations, which are then represented as subject–predicate–object triplets.

The repository includes both the final prepared `.spo` files and the source scripts used to generate them.

#### Knowledge Graph Files

```text
kg/
├── AristoTuple.spo
├── ConceptNet.spo
└── WordNet.spo
```

#### Knowledge Graph Preparation Scripts

```text
src/knowledge_graphs_preparation/
├── AristoTuple_to_spo.py
├── ConceptNet_to_spo.py
└── WordNet_to_spo.py
```

### Dataset Collection and Preprocessing

The experiments use six NLI datasets: **RTE**, **SICK**, **QNLI**, **DNLI**, **SNLI**, and **SciTail**. Dataset preparation followed two main procedures, depending on whether each dataset was directly supported by the `jiant` framework.

For datasets supported by `jiant`, the original non-enriched datasets can be downloaded automatically using the built-in `download_data` utility. These datasets are placed under the `tasks/data/` directory, following the structure expected by the framework.

```python
EXP_DIR = "/path/to/your/workspace/name_workspace/kg-nli-bert"
import sys
sys.path.insert(0, f"{EXP_DIR}/jiant")
import jiant.scripts.download_data.runscript as downloader

# Example: download a jiant-supported dataset
downloader.download_data(["qnli"], f"{EXP_DIR}/tasks")
```

For datasets that are not directly supported by `jiant`, the original files were downloaded manually from their official sources and then processed into the format required by the framework. Although **SciTail** is supported by `jiant`, it was additionally preprocessed in this work in order to support the knowledge-enrichment process and produce the enriched SciTail dataset configuration. The preprocessing scripts are provided in:

```text
src/datasets_preprocessing/
├── dnli_preprocessing.py
├── sick_preprocessing.py
└── scitail_transform.py
```

After preprocessing, these datasets are also placed under `tasks/data/`, together with the datasets downloaded automatically through `jiant`.

### Dataset Enrichment with Knowledge Graph Triplets

After the original no-enriched datasets and the knowledge graph files had been prepared, the next step was to connect the dataset examples with relevant external knowledge. For each sentence pair, the enrichment process examines the sentence text through **n-grams**, meaning continuous sequences of one or more words. These n-grams are used to identify possible concepts that also appear as subjects in the knowledge graph. When a matching concept is found, candidate triplets connected to that subject are retrieved from the graph.

The retrieved triplets are not added automatically. Each candidate triplet is compared with the original sentence using sentence embeddings from `paraphrase-MiniLM-L6-v2` and cosine similarity. Only the most two relevant triplets, according to a predefined similarity threshold, are appended to the sentence. This process produces knowledge-enriched versions of the training sets. The validation and test sets remain unchanged, so that the final evaluation measures how well the model generalizes from enriched training examples to unknown, non-enriched examples.

The code used for this step is provided in:

```text
src/enrichment/
└── dataset_enrichment.py
```

The enriched training files are saved under `tasks/data/`, together with the original datasets, so that they can be used by the same `jiant` pipeline for tokenization, caching, training, and evaluation.

### Jiant Task Integration

After the knowledge-enriched training files were created, the enriched datasets had to be connected with the `jiant` framework so that they could be used as regular tasks during tokenization, caching, training, and evaluation.

This step does not require uploading or modifying the entire `jiant` framework in this repository. Instead, only the files that were changed or added are provided under the `modified_jiant/` directory. These files are copied into the corresponding locations of the cloned `jiant` repository during setup.

```text
modified_jiant/
├── retrieval.py
├── core.py
└── tasks/
    └── lib/
        ├── qnli_enriched.py
        ├── rte_enriched.py
        ├── sick_enriched.py
        ├── dnli_enriched.py
        ├── snli_enriched.py
        ├── scitail_enriched.py
        └── ...
```

Each file under `modified_jiant/tasks/lib/` defines an enriched task. These task files specify how the corresponding dataset is loaded, how the training, validation, and test examples are read, which labels are used, and how each example is converted into the internal format expected by `jiant`.

The `retrieval.py` file registers the enriched task names in the `jiant` task dictionary. This allows the enriched datasets to be called by name from the configuration files, such as `qnli_enriched`, `rte_enriched`, `sick_enriched`, `dnli_enriched`, `snli_enriched`, and `scitail_enriched`.

The `core.py` file connects the enriched tasks with the evaluation procedure used in the experiments. Since the NLI tasks are classification tasks, accuracy is used as the evaluation metric.

To apply the modified `jiant` files, run the following commands from the main project directory:

```bash
# Replace the modified task retrieval file
cp modified_jiant/retrieval.py jiant/jiant/tasks/retrieval.py

# Replace the modified evaluation file
cp modified_jiant/core.py jiant/jiant/tasks/evaluate/core.py

# Copy the enriched task definition files into jiant
cp -rn modified_jiant/lib/* jiant/jiant/tasks/lib/
```

Configuration files are stored separately from the modified `jiant` code under:

```text
tasks/config/
```

These configuration files define the task name and the paths to the corresponding training, validation, and test files.

### Training and Evaluation with Jiant

After the enriched tasks were integrated into `jiant`, the next step was to train and evaluate the BERT-based models on the original and knowledge-enriched dataset configurations. This process is handled by the main `jiant` execution script provided in:

```text
src/training/
└── run_jiant.py
```

This script brings together the main stages required by the `jiant` workflow. It first defines the project paths and loads the necessary `jiant` modules. The two pretrained Hugging Face models are then exported into a format compatible with `jiant`, so that it can be used during training.

Next, the selected task is tokenized and cached for the train, validation, and test phases. The cached files are used to speed up training and evaluation, avoiding repeated preprocessing of the same data. A small sample from the cached data can also be inspected to verify that the input ids, tokens, and labels were created correctly.

The script then creates the run configuration, where the task name, cached data path, batch sizes, number of epochs, number of GPUs, and other experiment hyperparameters are defined. Finally, the `jiant` run loop is executed to train the model, evaluate it on the validation set, and generate test predictions.

In this work, the same execution pipeline was used for both original and knowledge-enriched dataset configurations. The main changes between experiments are the selected task name, the corresponding configuration file, the result paths, the hyperparameters and the model checkpoint used.

### Final Test Evaluation

After training and evaluation with `jiant`, the model predictions for the test set are stored in a `test_preds.p` file. These predictions are used in the final evaluation stage. Since not all benchmark datasets provide publicly available test labels, two different evaluation procedures are included in this repository.

```text
src/evaluation/
├── local_test_accuracy.py
└── glue_submission.py
```

The `local_test_accuracy.py` script is used for datasets whose gold-standard test labels are available. It loads the original test file, extracts the true labels, converts them into the same numerical format as the model predictions, and loads the stored predictions from `test_preds.p`. After checking that the number of predictions matches the number of test examples, the script computes the final test accuracy locally.

The `glue_submission.py` script is used for datasets whose official test labels are not publicly released, such as GLUE benchmark test sets. In this case, local accuracy cannot be computed directly. Instead, the stored predictions are converted into the official GLUE submission format. The script extracts the prediction values and their corresponding example identifiers, maps the numerical predictions back to the required textual labels, restores the original test indices, and saves the output as a `.tsv` file with the required columns. The generated `.tsv` file can then be submitted to the official GLUE evaluation server in order to obtain the final test accuracy score for the corresponding task.

### Results

The following table summarizes the validation and test accuracy scores obtained for the baseline BERT model and the knowledge-enriched configurations. The reported values correspond to accuracy (%).

| Model / Dataset | RTE Val | RTE Test | SICK Val | SICK Test | QNLI Val | QNLI Test | DNLI Val | DNLI Test | SNLI Val | SNLI Test | SciTail Val | SciTail Test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BERT-base-uncased | 83.75 | 77.30 | 87.80 | 87.76 | 90.95 | 90.40 | 89.01 | 90.21 | 90.80 | 90.48 | 94.32 | 92.43 |
|+AristoTuple | **84.83** | 76.00 | 88.40 | **87.78** | **91.68** | **90.80** | 89.29 | 90.59 | 91.01 | **90.71** | **95.24** | 93.37 |
| +WordNet  | 84.11 | **77.50** | 88.60 | 87.62 | 91.56 | 90.70 | **89.41** | 90.82 | 90.70 | 90.68 | 94.63 | 93.41 |
| +ConceptNet  | — | — | — | — | — | — | — | — | 90.84 | 90.69 | 94.78 | **93.70** |
