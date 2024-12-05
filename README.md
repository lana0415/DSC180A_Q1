# Hourly Wage for Babysitters ChatGBT-4o-mini Audit

`Environment.yml` required package dependencies

Folders:
- `wage_audit` folder contains necessary notebooks and files to complete an audit
   - `input_data` contains jsonl files to be submited to BatchWizard for processing
   - `output_data` contains resulting files from BatchWizard
   - `processed_data` contains parsed dataframe
   - `q1_checkpoint/luigi_permute` q1 checkpoint audit on Ollama
   - `src` contains needed python files

Notebooks:
   - `step1_prompt_bulk_generator.ipynb` generates prompt file ready for submission (no need to use unless generating own data)`
   - `step2_parse_clean_data_babysitter.ipynb` parses output data ready for data analysis
   - `step3_data_analysis.ipynb` contains EDA, hypothesis testing, and regression
