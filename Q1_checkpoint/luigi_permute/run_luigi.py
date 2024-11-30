#!/usr/bin/env python3
# coding: utf-8

import luigi
import threading
import time
import sqlite3
import pandas as pd
import yaml
import os
import re
import sys
import logging
from tqdm.auto import tqdm
import ollama
from ollama import Options
import argparse
from datetime import datetime
import itertools

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
        
    if not verbose:
        luigi.interface.InterfaceLogging.setup(type('opts',
                                            (),
                                            {   'background': None,
                                                'logdir': None,
                                                'logging_conf_file': None,
                                                'log_level': 'INFO' 
                                            }))
        

def get_db_connection(db_file):
    conn = sqlite3.connect(db_file, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('PRAGMA journal_mode=WAL;')
    conn.commit()
    return conn

def setup_database(db_file, table_name):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    
    conn = get_db_connection(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TEXT,
            experiment_name TEXT,
            model_name TEXT,
            provider TEXT,
            trial INTEGER,
            prompt TEXT,
            response TEXT,
            major TEXT,
            gender_p TEXT,
            school TEXT,
            PRIMARY KEY (model_name, trial, major, gender_p, school)
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

class DatabaseTarget(luigi.Target):
    def __init__(self, db_file, table_name, model_name, trial, condition):
        self.db_file = db_file
        self.table_name = table_name
        self.model_name = model_name
        self.trial = trial
        self.condition = condition

    def exists(self):
        conn = get_db_connection(self.db_file)
        cursor = conn.cursor()
        cursor.execute(
            f'''
            SELECT 1 FROM {self.table_name}
            WHERE model_name = ? AND trial = ? AND major = ? AND gender_p = ? AND school = ?
            ''',
            (
                self.model_name,
                self.trial,
                self.condition['major'],
                self.condition['gender_p'],
                self.condition['school']
            )
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None

class RunExperimentTask(luigi.Task):
    config_file = luigi.Parameter()
    model = luigi.DictParameter()
    condition = luigi.DictParameter()
    trial = luigi.IntParameter()
    replace = luigi.BoolParameter(default=False)

    def requires(self):
        return LoadConfig(self.config_file)
    
    def query_model(self, prompt, provider, model_name):
        if provider == 'ollama':
            message = [{
                        'role': 'user',
                        'content': prompt,
                     }]

            start_time = time.time()
            response = ollama.chat(model=model_name, messages=message, options=Options(temperature=0.0, num_predict=150))
            response_content = response['message']['content']
            response_time = time.time() - start_time
            return response_content, response_time

    def output(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        db_file = f"data/{experiment_name}/results.db"
        table_name = config.get('table_name', 'results')
        return DatabaseTarget(
            db_file=db_file,
            table_name=table_name,
            model_name=self.model['name'],
            trial=self.trial,
            condition=self.condition
        )

    def run(self):
        config = self.load_config()
        prompt_template = config['prompt']
        prompt = prompt_template.format(**self.condition)
        model_name = self.model['name']
        provider = self.model['provider']
        experiment_name = config['experiment_name']
        table_name = config.get('table_name', 'results')
        timestamp = datetime.utcnow().isoformat()

        db_file = self.get_db_file()
        conn = get_db_connection(db_file)
        cursor = conn.cursor()

        # Check if entry exists (handled by output().exists())
        if not self.replace and self.output().exists():
            logging.info(f"Entry already exists for model '{model_name}', trial {self.trial}, condition {self.condition}. Skipping.")
            return

        try:
            response_content, response_time = self.query_model(prompt, provider, model_name)
        except Exception as e:
            logging.error(f"Error in query: {e}")
            response_content = f"Error: {e}"

        # Prepare data for database
        data = (
            timestamp, experiment_name, model_name, provider, self.trial,
            prompt, response_content,
            self.condition['major'], self.condition['gender_p'], self.condition['school']
        )

        # Insert into database
        with DB_LOCK:
            insert_query = f'''
            INSERT OR IGNORE INTO {table_name} (
                timestamp, experiment_name, model_name, provider, trial, prompt, response,
                major, gender_p, school
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            cursor.execute(insert_query, data)
            conn.commit()

        conn.close()

    def get_db_file(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        return f"data/{experiment_name}/results.db"

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

class ExperimentWrapper(luigi.WrapperTask):
    config_file = luigi.Parameter()
    replace = luigi.BoolParameter(default=False)

    def requires(self):
        config = self.load_config()
        models = config['models']
        num_trials = config['num_trials_per_condition']
        conditions_list = self.generate_conditions(config['conditions'])
        setup_database(self.get_db_file(), config.get('table_name', 'results'))

        tasks = []
        for model in models:
            for condition in conditions_list:
                for trial in range(num_trials):
                    tasks.append(
                        RunExperimentTask(
                            config_file=self.config_file,
                            model=model,
                            condition=condition,
                            trial=trial,
                            replace=self.replace
                        )
                    )
        return tasks

    def generate_conditions(self, conditions_dict):
        # Generate all combinations of conditions
        condition_keys = []
        condition_values = []

        for condition in conditions_dict:
            for key, values in condition.items():
                condition_keys.append(key)
                condition_values.append(values)

        combinations = list(itertools.product(*condition_values))

        conditions_list = []
        for combo in combinations:
            condition = dict(zip(condition_keys, combo))
            conditions_list.append(condition)
        return conditions_list

    def get_db_file(self):
        config = self.load_config()
        experiment_name = config.get('experiment_name', 'default_experiment')
        return f"data/{experiment_name}/results.db"

    def load_config(self):
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

def main():
    parser = argparse.ArgumentParser(description="Experiment Runner with Luigi")
    parser.add_argument("config_file", help="Path to the experiment configuration YAML file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--replace", action="store_true", help="Replace existing experiment database entries")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Set global database file
    global DATABASE_FILE
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    experiment_name = config.get('experiment_name', 'default_experiment')
    DATABASE_FILE = f"data/{experiment_name}/results.db"

    # Use num_gpu from the config file for the number of workers
    num_gpu = config.get('num_gpu', 1)

    # Run Luigi pipeline
    try:
        luigi.build(
            [ExperimentWrapper(config_file=args.config_file, replace=args.replace)],
            workers=num_gpu,
            local_scheduler=True,
            no_lock=True  # Added parameter to help with interrupt handling
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
