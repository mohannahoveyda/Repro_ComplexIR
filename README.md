# Reproducing Set-Compositional Complex Information Retrieval 
This is the repository for the reproducibility paper "Reproducing Set-Compositional Complex Information Retrieval". Here you can find all the necessary scripts for replicating the results of the various models used. 

## Install & Dependencies

`faiss`

`torch`

Installation: 

```conda create -n retrieval_cpu python=3.10 pytorch torchvision torchaudio cpuonly faiss-cpu pandas -c pytorch -c conda-forge -y```

and then 

```pip install psutil flash_attn sentence-transformers==4.1.0 transformers==4.43.1 ```

Had version conflicts with transformer and sentence-transformers, this is only the cpu version install! 

```conda create -n retrieval_gpu python=3.10 pytorch torchvision torchaudio faiss-cpu pandas -c pytorch -c conda-forge -y```

change `faiss-cpu` to `faiss-gpu` for faster indexing. But see https://github.com/facebookresearch/faiss/blob/main/INSTALL.md for further details.



## Run Experiments 


## Evaluation

First ensure the qrels file (QUEST or LIMIT) and the run file are in the desired TREC format:

```bash
python code/evaluation/preprocess_qrels.py --input data/QUEST/test.jsonl 
```

```bash
python code/evaluation/evaluate.py --run outputs/runs/toy_run_trec --qrels data/QUEST/test_qrels --cutoffs 1 5 10

 ```