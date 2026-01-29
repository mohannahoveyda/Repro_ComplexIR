# How to run:

## Whole data

For LLama-70B
```bash
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="70B_llama_instr_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_llama3_70B_instr.slurm

```


For Llama-8B
```bash 
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="8B_llama_instr_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_llama3_8B_instr.slurm
```


## Samples:

for llama-8b-sample
```bash 
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="Meta-Llama-3-8B-Instruct_sample_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_llama3_8B_instruct_sample.slurm
```

for llama-70b-sample
```bash 
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="Llama-3.3-70B-Instruct_sample_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_llama3_70B_instruct_sample.slurm
```


for olmo-7B-sample
```bash 
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="OLMo-2-1124-7B-Instruct_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_olmo_2_7B_instruct.slurm
```

for olmo-32B-sample
```bash 
export TIMESTAMP=$(date +'%Y%m%d_%H-%M-%S')
sbatch --job-name="OLMo-2-0325-32B-Instruct_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_olmo_2_32B_instruct.slurm
```

for mistral-7B-sample
```bash
sbatch --job-name="Mistral-7B-Instruct-v0.1_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_mistral_7B_instruct_v1.slurm
```

for mistral-8x7B-sample
```bash
sbatch --job-name="Mixtral-8x7B-Instruct-v0.1_${TIMESTAMP}" --export=ALL,TIMESTAMP="${TIMESTAMP}" /home/mhoveyda1/REASON/slurm_scripts/run_mistral_8x7B_instruct_v1.slurm
```

