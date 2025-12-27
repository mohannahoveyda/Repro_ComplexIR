# Reproducing Set-Compositional Complex Information Retrieval 
This is the repository for the reproducibility paper "Reproducing Set-Compositional Complex Information Retrieval". Here you can find all the necessary scripts for replicating the results of the various models used. 

## Install & Dependencies

## Run Experiments 


## Evaluation

First ensure the qrels file (QUEST or LIMIT) and the run file are in the desired TREC format:

```bash
python code/evaluation/preprocess_qrels.py --input data/QUEST/test.jsonl 
```

```bash
python code/evaluation/evaluate.py --run outputs/runs/toy_run_trec --qrels data/QUEST/test_qrels --cutoffs 1 5 10

 ```