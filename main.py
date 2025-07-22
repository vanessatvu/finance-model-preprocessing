import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chunking import load_tokenizer, chunk_and_classify_text
from config import INPUT_FILE, OUTPUT_FILE
from utils import save_to_jsonl, save_to_txt  # include save_to_txt
from megaparse import merge_similar_chunks

def main():
    tokenizer = load_tokenizer()
    chunks = chunk_and_classify_text(INPUT_FILE, tokenizer)
    save_to_jsonl(chunks, OUTPUT_FILE)
    
    # Save as .txt (same name, different extension)
    txt_output = OUTPUT_FILE.replace(".jsonl", ".txt")
    save_to_txt(chunks, txt_output)

    # run megaparse and merge similar chunks
    merged = merge_similar_chunks(chunks)
    save_to_jsonl(merged, "output/final_merged_output.jsonl")
    save_to_txt(merged, "output/final_merged_output.txt")

if __name__ == "__main__":
    main()

