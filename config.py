# config.py

# load model name to match tokenizer and future fine-tuning model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # can change later

# set thresholds and parameters
TOKEN_THRESHOLDS = {
    "light": 100,
    "medium": 300  # >300 = heavy
}

MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64  # optional context overlap between chunks

# input/output file paths
INPUT_FILE = "data/combined_output.txt"
OUTPUT_FILE = "output/classified_chunks.jsonl"

