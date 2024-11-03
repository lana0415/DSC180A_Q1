- `ollama pull llama2:7b-chat-q4_K_M`
- `ollama pull llama3:8b-instruct-q4_K_M`
- `pip install -r requirements.txt`
- `python3 run_luigi.py salary.yml`
  - for verbose debugging: `python3 run_luigi.py --verbose salary.yml`
  - to replace existing output file (specified in yml), otherwise will resume: `python3 run_luigi.py --replace salary.yml`