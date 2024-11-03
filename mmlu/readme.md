# How to run

## Setup
1. Install [ollama](https://ollama.com), no need to download models
1. `pip install -r requirements.txt`
1. Edit yml to set `num_gpu` to number of GPUs (set to 1 for Apple M or CPU only)


## To run

`python3 run_mmlu_multigpu.py [-v] mmlu_full_run.yml`

`python3 run_mmlu_multigpu.py [-v] mmlu_n500_run.yml`
