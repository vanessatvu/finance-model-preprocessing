import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import INPUT_FILE, OUTPUT_FILE
from utils import save_to_jsonl, save_to_txt  
from chunking import load_tokenizer, chunk_and_classify_text
from chunk_merger import merge_similar_chunks, semantic_rechunk_pass

def main():
    tokenizer = load_tokenizer()

    # initial chunking
    chunks = chunk_and_classify_text(INPUT_FILE, tokenizer)
    save_to_jsonl(chunks, OUTPUT_FILE)
    save_to_txt(chunks, OUTPUT_FILE.replace(".jsonl", ".txt"))

    # 1st pass: merged output
    merged = merge_similar_chunks(chunks)
    save_to_jsonl(merged, "output/final_merged_output.jsonl")
    save_to_txt(merged, "output/final_merged_output.txt")

    # 2nd pass: semantic rechunking
    final_chunks = semantic_rechunk_pass(merged)
    save_to_jsonl(final_chunks, "output/final_rechunked_output.jsonl")
    save_to_txt(final_chunks, "output/final_rechunked_output.txt")

if __name__ == "__main__":
    main()