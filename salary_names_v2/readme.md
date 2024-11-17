# Salary Negotiation - Names x Major audit

* `step1_prompt_bulk_generator.ipynb` inputs `input_data\audit_names.xlsx` and generates prompts in OpenAI Batch jsonl format. These must be submitted to the [OpenAI Batch API](https://platform.openai.com/batches) or for open-weight models, can be submitted with [vllm batch](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference_openai.md). They are large but easily compressible, so they are stored as .zip in GitHub. To replicate this with OpenAI or vllm, you must unzip these and only submit the raw .jsonl file.
* OpenAI or vllm returns a result .jsonl file. These must be placed in `input_data\batch_results`.
* `step2_parse_clean_data.ipynb` reads all .jsonl or .jsonl.zip files in `input_data\batch_results`, parses the dollar value from the prompt, and stores all results for all models in `processed_data/emp_name_major_allmodels.csv.zip`
* `step3_analysis.ipynb` reads `processed_data/emp_name_major_allmodels.csv.zip` and does the analyses.
