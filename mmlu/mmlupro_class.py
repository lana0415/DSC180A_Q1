#!/usr/bin/env python3
# coding: utf-8

import argparse
import threading
import time
import signal
import sqlite3
import pandas as pd
import yaml
import traceback
import os
import re
import sys
import logging
from tqdm.auto import tqdm
import concurrent.futures
from huggingface_hub import hf_hub_download
import ollama  # Ensure this is installed
from ollama import Options


class ExperimentRunner:
    def __init__(self, config_file, verbose=False, resume=False):
        self.verbose = verbose
        self.resume = resume
        self.terminate_flag = False
        self.batched_inserts = []
        self.db_lock = threading.Lock()
        self.config_file = config_file
        self.completed_models = 0
        self.total_models = 0

        # Initialize logging
        self.setup_logging()

        # Load configuration
        self.load_config()

        # Initialize database connections
        self.database_file = f"data/{self.experiment_name}.db"
        os.makedirs(os.path.dirname(self.database_file), exist_ok=True)
        self.disk_conn = None

        # Progress bar position management
        self.position_lock = threading.Lock()
        self.current_position = 0

        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
        )
        # Suppress logs from third-party libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('ollama').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)

    def signal_handler(self, sig, frame):
        logging.info("\nExecution interrupted by user. Flushing data and exiting gracefully...")
        self.terminate_flag = True

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            self.experiment_name = self.config.get('experiment_name', 'default_experiment')
            self.dataset_info = self.config['dataset']
            self.models = self.config['models']
            self.num_gpu = self.config.get('num_gpu', 1)
            self.pre_conditions = self.config['pre_conditions']
            self.post_conditions = self.config['post_conditions']
            self.subset = self.dataset_info.get('subset', None)
            self.table_name = self.config.get('table_name', 'results')
            logging.debug(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def run(self):
        try:
            # Check if the database exists and whether to resume or replace
            self.check_existing_db()

            # Load dataset
            self.df_all = self.load_dataset()

            # Initialize database connections
            self.initialize_database()

            # Initialize overall progress bar
            self.total_models = len(self.models)
            self.tqdm_outer = tqdm(total=self.total_models, desc="Total Models Processed")

            # Start flushing thread
            self.flush_thread = threading.Thread(target=self.periodic_flushing)
            self.flush_thread.start()

            # Run models in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpu) as executor:
                futures = []
                for model in self.models:
                    futures.append(executor.submit(self.process_model, model))
                for future in concurrent.futures.as_completed(futures):
                    pass  # Handle exceptions here if needed

            # Final flush
            self.flush_data_to_disk()

            # Close connections
            self.disk_conn.close()

            # Wait for threads
            self.terminate_flag = True
            self.flush_thread.join()

            self.tqdm_outer.close()

            logging.info("Experiment completed.")

        except Exception as e:
            logging.error(f"An error occurred in run(): {e}")
            traceback.print_exc()

    def check_existing_db(self):
        if os.path.exists(self.database_file):
            if self.resume:
                logging.info(f"Resuming experiment using existing database '{self.database_file}'.")
            else:
                os.remove(self.database_file)
                logging.info(f"Database '{self.database_file}' replaced.")
        else:
            if self.resume:
                logging.error(f"No existing database found to resume. Starting a new experiment.")
            else:
                logging.info(f"Starting a new experiment. Database will be created at '{self.database_file}'.")

    def load_dataset(self):
        try:
            REPO_ID = self.dataset_info['repo_id']
            SUBFOLDER = self.dataset_info.get('subfolder', '')
            FILENAME = self.dataset_info['filename']

            logging.info(f"Loading dataset from repo: {REPO_ID}, subfolder: {SUBFOLDER}, filename: {FILENAME}")

            dataset_path = hf_hub_download(repo_id=REPO_ID, subfolder=SUBFOLDER, filename=FILENAME, repo_type="dataset")
            df = pd.read_parquet(dataset_path)
            if self.subset is not None and isinstance(self.subset, int):
                df = df.head(self.subset)
            logging.info("Dataset loaded successfully.")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            sys.exit(1)

    def initialize_database(self):
        self.disk_conn = sqlite3.connect(self.database_file, check_same_thread=False)
        self.create_table(self.disk_conn)

    def create_table(self, conn):
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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

    def periodic_flushing(self):
        while not self.terminate_flag:
            time.sleep(60)
            if self.terminate_flag:
                break
            self.flush_data_to_disk()

    def flush_data_to_disk(self):
        if self.batched_inserts:
            with self.db_lock:
                cursor = self.disk_conn.cursor()
                cursor.executemany(f'''
                    INSERT OR IGNORE INTO {self.table_name} (
                        question_id, pre_cond_desc, post_cond_desc, subject, question, options, 
                        correct_answer, model_name, model_response, model_output, model_input, response_time_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', self.batched_inserts)
                self.disk_conn.commit()
                self.batched_inserts.clear()

    def process_model(self, model):
        with self.position_lock:
            position = self.current_position
            self.current_position += 1

        model_name = model['name']
        provider = model.get('provider', 'ollama')
        total_tasks_for_model = len(self.pre_conditions) * len(self.post_conditions) * len(self.df_all)
        pbar = tqdm(total=total_tasks_for_model, desc=f"Model: {model_name}", position=position, leave=False)

        try:
            existing_entries = set()
            if self.resume:
                existing_entries = self.get_existing_entries(model_name)

            for pre_condition_data in self.pre_conditions:
                pre_cond_desc = pre_condition_data['descriptor']
                pre_cond_text = pre_condition_data['text']

                for post_condition_data in self.post_conditions:
                    post_cond_desc = post_condition_data['descriptor']
                    post_cond_text = post_condition_data['text']

                    for index, row in self.df_all.iterrows():
                        if self.terminate_flag:
                            pbar.close()
                            return

                        question_id = f"q{index+1}"
                        primary_key = (question_id, pre_cond_desc, post_cond_desc, model_name)
                        if primary_key in existing_entries:
                            pbar.update(1)
                            continue  # Skip already processed entries

                        try:
                            subject = row["category"]
                            question = row["question"]
                            options = row["options"]
                            correct_answer_index = row["answer_index"]
                            option_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                            correct_answer_letter = option_labels[correct_answer_index]

                            prompt = self.construct_prompt(pre_cond_text, subject, question, options, post_cond_text)
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
                                ', '.join(options),
                                correct_answer_letter,
                                model_name,
                                model_response,
                                model_output,
                                prompt,
                                response_time
                            )
                            with self.db_lock:
                                self.batched_inserts.append(result)
                        except Exception as e:
                            logging.error(f"Error processing question {question_id} for model '{model_name}': {e}")
                            traceback.print_exc()
                        finally:
                            pbar.update(1)

            pbar.close()
            with self.db_lock:
                self.completed_models += 1
            self.tqdm_outer.update(1)

        except Exception as e:
            logging.error(f"An error occurred in process_model() for model '{model_name}': {e}")
            traceback.print_exc()
        finally:
            pbar.close()

    def get_existing_entries(self, model_name):
        cursor = self.disk_conn.cursor()
        cursor.execute(f'''
            SELECT question_id, pre_cond_desc, post_cond_desc, model_name FROM {self.table_name}
            WHERE model_name = ?
        ''', (model_name,))
        existing_entries = set(cursor.fetchall())
        return existing_entries

    @staticmethod
    def extract_answer(text):
        match = re.search(r'\b([A-Ja-j])\)', text)
        if match:
            return match.group(1).upper()

        match = re.search(r'\b([A-Ja-j])\b(?=\W*$)', text.strip())
        if match:
            return match.group(1).upper()

        match = re.search(r'^\b([A-Ja-j])\b', text.strip())
        if match:
            return match.group(1).upper()

        return "INVALID"

    @staticmethod
    def construct_prompt(pre_condition, subject, question, options, post_condition):
        prompt = f"{pre_condition}\n\nSubject: {subject}\nQuestion: {question}\nOptions:\n"
        option_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        for i, option in enumerate(options):
            prompt += f"{option_labels[i]}) {option}\n"
        prompt += f"{post_condition}"
        return prompt

    @staticmethod
    def query_model(provider, model_name, prompt):
        try:
            if provider == 'ollama':
                response = ollama.generate(model=model_name, prompt=prompt, options=Options(temperature=0.0, num_predict=100))
                model_output = response['response']
                return model_output
            else:
                raise NotImplementedError(f"Provider '{provider}' is not implemented.")
        except Exception as e:
            logging.error(f"Error querying model '{model_name}': {e}")
            return ""


def main():
    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument('config_file', type=str, help='Path to the experiment configuration YAML file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume existing experiment')
    args = parser.parse_args()

    runner = ExperimentRunner(config_file=args.config_file, verbose=args.verbose, resume=args.resume)
    runner.run()


if __name__ == "__main__":
    main()
