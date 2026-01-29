# RSN

This is the repository of the "Reasoning for IR" project. 

# Reproducing The Experiments
To reproduce the experimets, you need to follow the steps below.


## 1. Preparing The Base Retrieval Results

### A. Download Datasets 

#### Quest
```bash 
# Download Quest
./helpers/download_quest_datasets.sh
export GOLD=./datasets/QUEST/test.jsonl
export GOLD_ID_ADDED=./datasets/QUEST/test_id_added.jsonl
python ./helpers/add_id_to_dataset.py $GOLD $GOLD_ID_ADDED
```

### B. Run The Base Retrievers (BM25 & E5-base)
```bash
export EXAMPLES=./datasets/QUEST/test_id_added.jsonl
export DOCS=./datasets/QUEST/tmp/documents.jsonl
export SAMPLE=0
export TOPK=20
```
#### Sparse: BM25
```bash
export MODEL=BM25
./run_retriever.sh
```
#### Dense: E5-base
```bash
export MODEL=E5
./run_retriever.sh
```

Now you should have two separate results files for BM25 and E5-base respectively. Each file contains the top-20 candidate entities for the each of the queries in QUEST. 

### C. Augmenting (retrieval) prediction files 
Running the following script will capture the metadata of retrieved and gold entities from Wikipedia and Wikidata (which we will need in further experiments later on). The resulting file is the augmented file with these metadata. 


```bash
./augment_retrieval_results.sh predictions/BM25/test_top20_sample0_2025-07-07_14-43.jsonl BM25
./augment_retrieval_results.sh predictions/E5/test_top20_sample0_2025-06-11_15-22.jsonl E5
```
## 2. Parsing Queries Beforehand

Run the following and once pass BM25 and once E5 to have all the necessary queries translated. They will be stored in cache for the later experiments. 

```bash 
sbatch slurms/parse_only.slurm 
```

## 3. Running Reasoning Modules

```bash
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
sbatch --export=TIMESTAMP /home/mhoveyda1/REASON/slurm_scripts/run_llama3_8B_instruct_sample.slurm
```




# 4. Evaluation
Evaluating baseline retrievers only

```bash
python src/main.py \
  --retriever BM25 \
  --baseline-only \
  --outdir results/baseline_retriever \
  --log_dir LOGS/baseline_retriever_evaluation

```



# 5. Exp
Running the experiments for BM25 base retriever and the model as meta-llama/Meta-Llama-3-8B-Instruct:

```bash 
sbatch slurms/run.slurm BM25 meta-llama/Meta-Llama-3-8B-Instruct
```


<!-- jid=$( sbatch --parsable --array=0-17 run.sh BM25 meta-llama/Meta-Llama-3-8B-Instruct ) -->

<!-- squeue -u $USER -h -o "%A %j %T %D" \
  | grep test.slurm -->


<!-- slurms/wait_and_merge.sh /home/mhoveyda1/REASON/runs/BM25/Meta-Llama-3-8B-Instruct_13079981 300 -->
