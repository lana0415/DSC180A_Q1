*reference main README

To Reproduce Results:
- download Ollama and ensure it can be accessed via command prompt
- copy full repository
- DELETE `results.db` located in `luigi_permute/data/salary_negotiation`
- cd into `luigi_permute` folder
- Run the following commands:
  - ollama pull llama2:7b-chat-q4_K_M
  - ollama pull llama3:8b-instruct-q4_K_M
  - pip install -r requirements.txt
  - python3 run_luigi_major.py major.yml
- To run demo on Salary Negotiation:
  - `python3 run_luigi.py salary.yml`

Notes:
- for verbose debugging: `python3 run_luigi.py --verbose salary.yml`
- to replace existing output file (specified in yml), otherwise will resume: `python3 run_luigi.py --replace salary.yml`
