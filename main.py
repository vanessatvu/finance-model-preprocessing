from chunking import load_tokenizer, chunk_and_classify_text
from config import INPUT_FILE, OUTPUT_FILE
from utils import save_to_jsonl, save_to_txt  # include save_to_txt

def main():
    tokenizer = load_tokenizer()
    chunks = chunk_and_classify_text(INPUT_FILE, tokenizer)
    save_to_jsonl(chunks, OUTPUT_FILE)
    
    # Save as .txt (same name, different extension)
    txt_output = OUTPUT_FILE.replace(".jsonl", ".txt")
    save_to_txt(chunks, txt_output)

    merged = merge_similar_chunks(chunks)
    save_to_txt(merged, "data/final_merged_output.txt")
    save_to_jsonl(merged, "data/final_merged_output.jsonl")

if __name__ == "__main__":
    main()

