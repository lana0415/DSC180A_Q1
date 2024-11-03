#!/usr/bin/env python3
# coding: utf-8

import luigi
import argparse
import threading
import time
import sqlite3
import pandas as pd
import glob
import yaml
import os
import re
import sys
import logging
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
import ollama 
from ollama import Options

# Global variables for database and logging
DATABASE_FILE = None
DB_LOCK = threading.Lock()
TERMINATE_FLAG = threading.Event()

def setup_logging(verbose):
    # Set up logging
    logging_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    # Suppress logs from third-party libraries
    for logger_name in ['urllib3', 'requests', 'httpx', 'ollama', 'asyncio', 'luigi']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def setup_database(db_file, table_name):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            question_id TEXT,
            pre_cond_desc TEXT,
            post_cond_desc TEXT,
            subject TEXT,
            question TEXT,
            options TEXT,
            correct_answer TEXT,
            model_name TEXT,
            model_response TEXT,
            model_output TEXT,
            model_input TEXT,
            response_time_seconds REAL,
            PRIMARY KEY (question_id, pre_cond_desc, post_cond_desc, model_name)
        )
    ''')
    conn.commit()
    conn.close()

class LoadConfig(luigi.Task):
    config_file = luigi.Parameter()

    def output(self):
        # No output file; the config is read directly
        return luigi.LocalTarget(self.config_file)

    def run(self):
        # Nothing to do since config is read directly
        pass

class LoadDataset(luigi.Task):
    config_file = luigi.Parameter()

    def requires(self):
        return LoadConfig(self.config_file)

    def output(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        dataset_path = f"data/{experiment_name}/dataset.parquet"
        return luigi.LocalTarget(dataset_path)

    def run(self):
        config = self.load_config()
        dataset_info = config['dataset']
        REPO_ID = dataset_info['repo_id']
        SUBFOLDER = dataset_info.get('subfolder', '')
        FILENAME = dataset_info['filename']
        subset = dataset_info.get('subset', None)

        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        logging.info(f"Loading dataset from repo: {REPO_ID}, subfolder: {SUBFOLDER}, filename: {FILENAME}")
        dataset_path = hf_hub_download(
            repo_id=REPO_ID, subfolder=SUBFOLDER, filename=FILENAME, repo_type="dataset"
        )
        df = pd.read_parquet(dataset_path)
        if subset is not None and isinstance(subset, int):
            df = df.head(subset)
        df.to_parquet(self.output().path)
        logging.info("Dataset loaded and saved.")

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

class ProcessModelTask(luigi.Task):
    config_file = luigi.Parameter()
    model_index = luigi.IntParameter()
    replace = luigi.BoolParameter(default=False)

    def requires(self):
        return LoadDataset(self.config_file)

    def output(self):
        # Output is determined by entries in the database, so we return an empty output
        return luigi.LocalTarget(f"data/temp/{self.get_model_name()}_complete.flag")

    def run(self):
        config = self.load_config()
        dataset_file = self.input().path
        df_all = pd.read_parquet(dataset_file)
        model = config['models'][self.model_index]
        model_name = model['name']
        provider = model.get('provider', 'ollama')
        pre_conditions = config['pre_conditions']
        post_conditions = config['post_conditions']
        table_name = config.get('table_name', 'results')

        db_file = self.get_db_file()
        conn = sqlite3.connect(db_file)
        conn.execute('pragma journal_mode=wal')
        cursor = conn.cursor()

        total_tasks = len(pre_conditions) * len(post_conditions) * len(df_all)
        pbar = tqdm(total=total_tasks, desc=f"Model: {model_name}", position=self.model_index, leave=False)

        try:
            for pre_cond in pre_conditions:
                if TERMINATE_FLAG.is_set():
                    break
                pre_cond_desc = pre_cond['descriptor']
                pre_cond_text = pre_cond['text']
                for post_cond in post_conditions:
                    if TERMINATE_FLAG.is_set():
                        break
                    post_cond_desc = post_cond['descriptor']
                    post_cond_text = post_cond['text']
                    for index, row in df_all.iterrows():
                        if TERMINATE_FLAG.is_set():
                            break
                        question_id = f"q{index+1}"
                        primary_key = (question_id, pre_cond_desc, post_cond_desc, model_name)
                        if self.entry_exists(cursor, table_name, primary_key):
                            pbar.update(1)
                            continue
                        try:
                            subject = row["category"]
                            question = row["question"]
                            options = row["options"]
                            correct_answer_index = row["answer_index"]
                            option_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                            correct_answer_letter = option_labels[correct_answer_index]
                            prompt = self.construct_prompt(
                                pre_cond_text, subject, question, options, post_cond_text
                            )
                            start_time = time.time()
                            model_output = self.query_model(provider, model_name, prompt)
                            response_time = time.time() - start_time
                            model_response = self.extract_answer(model_output)
                            result = (
                                question_id,
                                pre_cond_desc,
                                post_cond_desc,
                                subject,
                                question,
                                ", ".join(options),
                                correct_answer_letter,
                                model_name,
                                model_response,
                                model_output,
                                prompt,
                                response_time,
                            )
                            with DB_LOCK:
                                cursor.execute(
                                    f'''
                                    INSERT OR IGNORE INTO {table_name} (
                                        question_id, pre_cond_desc, post_cond_desc, subject, question, options,
                                        correct_answer, model_name, model_response, model_output, model_input, response_time_seconds
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''',
                                    result
                                )
                                conn.commit()
                        except Exception as e:
                            logging.error(
                                f"Error processing question {question_id} for model '{model_name}': {e}"
                            )
                        finally:
                            pbar.update(1)
        finally:
            pbar.close()
            conn.close()
            # Create an empty flag file to indicate completion
            os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
            with self.output().open('w') as f:
                f.write('Processing complete.')

    def entry_exists(self, cursor, table_name, primary_key):
        question_id, pre_cond_desc, post_cond_desc, model_name = primary_key
        cursor.execute(
            f'''
            SELECT 1 FROM {table_name}
            WHERE question_id = ? AND pre_cond_desc = ? AND post_cond_desc = ? AND model_name = ?
            ''',
            (question_id, pre_cond_desc, post_cond_desc, model_name)
        )
        return cursor.fetchone() is not None

    def query_model(self, provider, model_name, prompt):
        if TERMINATE_FLAG.is_set():
            return ""
        try:
            if provider == "ollama":
                response = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    options=Options(temperature=0.0, num_predict=100),
                )
                return response.get("response", "")
            else:
                raise NotImplementedError(f"Provider '{provider}' is not implemented.")
        except Exception as e:
            logging.error(f"Error querying model '{model_name}': {e}")
            return ""

    @staticmethod
    def extract_answer(text):
        for pattern in [r"\b([A-Ja-j])\)", r"\b([A-Ja-j])\b(?=\W*$)", r"^\b([A-Ja-j])\b"]:
            match = re.search(pattern, text.strip())
            if match:
                return match.group(1).upper()
        return "INVALID"

    @staticmethod
    def construct_prompt(pre_cond, subject, question, options, post_cond):
        prompt = f"{pre_cond}\n\nSubject: {subject}\nQuestion: {question}\nOptions:\n"
        for label, option in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], options):
            prompt += f"{label}) {option}\n"
        prompt += post_cond
        return prompt

    def get_db_file(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        return f"data/{experiment_name}/results.db"

    def get_model_name(self):
        config = self.load_config()
        model = config['models'][self.model_index]
        return model['name']

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

class ExperimentWrapper(luigi.WrapperTask):
    config_file = luigi.Parameter()
    replace = luigi.BoolParameter(default=False)

    def requires(self):
        config = self.load_config()
        num_models = len(config['models'])
        setup_database(self.get_db_file(), config.get('table_name', 'results'))

        tasks = []
        for i in range(num_models):
            tasks.append(
                ProcessModelTask(
                    config_file=self.config_file,
                    model_index=i,
                    replace=self.replace
                )
            )
        return tasks

    def get_db_file(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        return f"data/{experiment_name}/results.db"

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

def signal_handler(signum, frame):
    logging.info("Interrupt received. Setting termination flag.")
    TERMINATE_FLAG.set()

def main():
    parser = argparse.ArgumentParser(description="Experiment Runner with Luigi")
    parser.add_argument("config_file", help="Path to the experiment configuration YAML file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--replace", action="store_true", help="Replace existing experiment database")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Set global database file
    global DATABASE_FILE
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    experiment_name = config.get('experiment_name', 'default_experiment')
    DATABASE_FILE = f"data/{experiment_name}/results.db"

    files = glob.glob('data/temp/*')
    for f in files:
        os.remove(f)

    if args.replace and os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)
        logging.info(f"Database '{DATABASE_FILE}' replaced.")

    # Register signal handler for interrupt
    import signal
    signal.signal(signal.SIGINT, signal_handler)

    # Use num_gpu from the config file for the number of workers
    num_gpu = config.get('num_gpu', 1)

    luigi.build(
        [ExperimentWrapper(config_file=args.config_file, replace=args.replace)],
        workers=num_gpu, log_level="INFO",
        local_scheduler=True
    )

if __name__ == "__main__":
    main()
