#!/bin/bash

python3 generate_letters.py --model llama2:7b-chat-q4_K_M &
python3 generate_letters.py --model llama3:8b-instruct-q4_K_M &
python3 generate_letters.py --model llama3.1:8b-instruct-q4_K_M &
python3 generate_letters.py --model llama3.2:3b-instruct-q4_K_M &
python3 generate_letters.py --model gemma2:9b-instruct-q4_K_M &
python3 generate_letters.py --model qwen2.5:7b-instruct-q4_K_M  &
python3 generate_letters.py --model mistral:7b-instruct-v0.3-q4_K_M &
python3 generate_letters.py --model mistral:7b-instruct-q4_K_M &
