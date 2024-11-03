#!/usr/bin/env python3
# coding: utf-8

import yaml
import os
import sys
import sqlite3
import pandas as pd
import ollama  # Adjust as necessary for your model provider
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
import re
import time
import datetime
import threading
import concurrent.futures
import traceback
import signal
from ollama import Options

# Global variable for verbosity
VERBOSE = False

def extract_answer(text):
    match = re.search(r'\b([A-Da-d])\)', text)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'\b([A-Da-d])\b(?=\W*$)', text.strip())
    if match:
        return match.group(1).upper()
    
    match = re.search(r'^\b([A-Da-d])\b', text.strip())
    if match:
        return match.group(1).upper()
    
    return "INVALID"

def signal_handler(sig, frame):
    print("\nExecution interrupted by user. Exiting gracefully...")
    sys.exit(0)

def main():
    global VERBOSE
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Check for the verbosity flag
        if '-v' in sys.argv:
            VERBOSE = True
            sys.argv.remove('-v')

        # Check if experiment YAML file is provided
        if len(sys.argv) != 2:
            print("Usage: python experiment.py [-v] <experiment_config.yml>")
            sys.exit(1)
        
        experiment_config_file = sys.argv[1]
        if VERBOSE: print(f"Loading configuration from {experiment_config_file}")
        
        # Load experiment configuration
        with open(experiment_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if VERBOSE: print("Configuration loaded successfully.")
        
        experiment_name = config.get('experiment_name', 'default_experiment')
        dataset_info = config['dataset']
        models = config['models']
        num_gpu = config['num_gpu']
        pre_conditions = config['pre_conditions']
        post_conditions = config['post_conditions']
        subset = dataset_info.get('subset', None)
        
        if VERBOSE:
            print(f"Experiment Name: {experiment_name}")
            print(f"Models: {[model['name'] for model in models]}")
            print(f"Pre-conditions: {[pc['descriptor'] for pc in pre_conditions]}")
            print(f"Post-conditions: {[pc['descriptor'] for pc in post_conditions]}")
            print(f"Subset: {subset}")
        
        # Generate output filename without timestamp to allow resuming
        database_file = f"{experiment_name}.db"
        table_name = config.get('table_name', 'results')
        if VERBOSE: print(f"Database file: {database_file}")
        if VERBOSE: print(f"Table name: {table_name}")
        
        # Load dataset
        df_all = load_dataset(dataset_info)
        if VERBOSE: print(f"Dataset loaded with {len(df_all)} rows.")
        
        # Apply subset if specified
        if subset is not None and isinstance(subset, int):
            df_all = df_all.head(subset)
            if VERBOSE: print(f"Subset applied. Processing {len(df_all)} rows.")
        
        # Create a lock for thread-safe database operations
        db_lock = threading.Lock()
        
        # Create the database and table if they don't exist
        try:
            conn = sqlite3.connect(database_file, check_same_thread=False)
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
            if VERBOSE: print("Database and table initialized.")
        except Exception as e:
            print(f"Error initializing database: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Run models in parallel
        try:
            if VERBOSE: print("Starting model processing...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpu) as executor:
                futures = []
                with tqdm(total=len(models)*len(pre_conditions)*len(post_conditions)) as pbar:
                    for model in models:
                        if VERBOSE: print(f"Submitting model {model['name']} to the executor.")
                        futures.append(executor.submit(process_model, model, df_all, pre_conditions, post_conditions, database_file, table_name, db_lock, pbar))
                    
                    # Wait for all models to finish
                    concurrent.futures.wait(futures)
                    pbar.close()
            if VERBOSE: print("Model processing completed.")
        except Exception as e:
            print(f"Error during model processing: {e}")
            traceback.print_exc()
        
        print("Experiment completed.")
    except Exception as e:
        print(f"An error occurred in main(): {e}")
        traceback.print_exc()

def process_model(model, df_all, pre_conditions, post_conditions, database_file, table_name, db_lock, pbar):
    global VERBOSE
    try:
        model_name = model['name']
        provider = model.get('provider', 'ollama')  # Default to 'ollama' if not specified
        if VERBOSE: print(f"Processing model: {model_name}")
        
        # Create a separate database connection for this thread
        conn = sqlite3.connect(database_file, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL;')
        cursor = conn.cursor()
        
        # Check if model exists in Ollama; if not, pull it
        if not check_model_exists(provider, model_name):
            pull_model(provider, model_name)
        
        # Iterate over pre_conditions and post_conditions independently
        for pre_condition_data in pre_conditions:
            pre_condition_text = pre_condition_data['text']
            pre_cond_desc = pre_condition_data['descriptor']
            if VERBOSE: print(f"Processing pre_condition: {pre_cond_desc}")
            
            for post_condition_data in post_conditions:
                post_condition_text = post_condition_data['text']
                post_cond_desc = post_condition_data['descriptor']
                if VERBOSE: print(f"Processing post_condition: {post_cond_desc}")
                
                # Unique descriptor for this combination
                instruction_descriptor = f"{pre_cond_desc} x {post_cond_desc}"
                
                # Process prompts sequentially for this model
                for index, row in tqdm(df_all.iterrows(), total=len(df_all), desc=f"Processing {model_name}: {instruction_descriptor}", leave=False):
                    try:
                        question_id = f"q{index+1}"  # Assign a unique question_id to each question
                        
                        # Check if this combination has already been processed
                        with db_lock:
                            cursor.execute(f'''
                                SELECT 1 FROM {table_name} 
                                WHERE question_id=? AND pre_cond_desc=? AND post_cond_desc=? AND model_name=?
                            ''', (question_id, pre_cond_desc, post_cond_desc, model_name))
                            if cursor.fetchone():
                                # Skip processing as it's already done
                                if VERBOSE: print(f"Model '{model_name}': Skipping question {question_id} (already processed).")
                                continue
                        
                        subject = row["subject"]
                        question = row["question"]
                        options = row["choices"]
                        correct_answer_index = row["answer"]
                        option_labels = ["A", "B", "C", "D"]
                        correct_answer_letter = option_labels[correct_answer_index]
                        
                        # Construct the prompt
                        prompt = construct_prompt(pre_condition_text, subject, question, options, post_condition_text)
                        
                        # Record start time
                        start_time = time.time()
                        
                        # Query the model
                        model_output = query_model(provider, model_name, prompt)
                        
                        # Record end time and calculate duration
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Extract the model response (e.g., extract answer letter)
                        model_response = extract_answer(model_output)
                        
                        # Store the result
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
                        
                        # Insert result into the database
                        with db_lock:
                            cursor.execute(f'''
                                INSERT OR IGNORE INTO {table_name} (
                                    question_id, pre_cond_desc, post_cond_desc, subject, question, options, 
                                    correct_answer, model_name, model_response, model_output, model_input, response_time_seconds
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', result)
                            conn.commit()
                        
                        if VERBOSE: print(f"Model '{model_name}': Processed question {question_id} with instruction '{instruction_descriptor}'")
                    except Exception as e:
                        print(f"Error processing question {question_id} for model '{model_name}': {e}")
                        traceback.print_exc()
                pbar.update(1)
        # Close the database connection for this thread
        conn.close()
    except Exception as e:
        print(f"An error occurred in process_model() for model '{model['name']}': {e}")
        traceback.print_exc()

def load_dataset(dataset_info):
    try:
        REPO_ID = dataset_info['repo_id']
        SUBFOLDER = dataset_info.get('subfolder', '')
        FILENAME = dataset_info['filename']
        
        if VERBOSE: print(f"Loading dataset from repo: {REPO_ID}, subfolder: {SUBFOLDER}, filename: {FILENAME}")
        
        # Load dataset
        dataset_path = hf_hub_download(repo_id=REPO_ID, subfolder=SUBFOLDER, filename=FILENAME, repo_type="dataset")
        df = pd.read_parquet(dataset_path)
        if VERBOSE: print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

def construct_prompt(pre_condition, subject, question, options, post_condition):
    try:
        prompt = f"{pre_condition}\n\nSubject: {subject}\nQuestion: {question}\nOptions:\n"
        option_labels = ["A", "B", "C", "D"]
        for i, option in enumerate(options):
            prompt += f"{option_labels[i]}) {option}\n"
        prompt += f"{post_condition}"
        return prompt
    except Exception as e:
        print(f"Error constructing prompt: {e}")
        traceback.print_exc()
        return ""

def query_model(provider, model_name, prompt):
    try:
        if provider == 'ollama':
            response = ollama.generate(model=model_name, prompt=prompt, options=Options(temperature=0.0, num_predict=100))
            model_output = response['response']
            return model_output
        else:
            raise NotImplementedError(f"Provider '{provider}' is not implemented.")
    except Exception as e:
        print(f"Error querying model '{model_name}': {e}")
        traceback.print_exc()
        return ""

def check_model_exists(provider, model_name):
    try:
        if provider == 'ollama':
            models_list = ollama.list()
            model_names = [model['name'] for model in models_list.get('models', [])]
            return model_name in model_names
        else:
            raise NotImplementedError(f"Provider '{provider}' is not implemented.")
    except Exception as e:
        print(f"Error checking if model '{model_name}' exists: {e}")
        traceback.print_exc()
        return False

def pull_model(provider, model_name):
    try:
        if provider == 'ollama':
            print(f"Model '{model_name}' not found in Ollama. Pulling the model...")
            ollama.pull(model_name)
            print(f"Model '{model_name}' pulled successfully.")
        else:
            raise NotImplementedError(f"Provider '{provider}' is not implemented.")
    except Exception as e:
        print(f"Failed to pull model '{model_name}': {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
