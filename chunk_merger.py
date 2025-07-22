from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# CONFIGS
SIMILARITY_THRESHOLD = 0.70
MIN_LENGTH_TO_CHECK = 10     

# Check if 2 chunks are related based on cosine similarity
def chunks_are_related(text1: str, text2: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    if len(text1.strip().split()) < MIN_LENGTH_TO_CHECK or len(text2.strip().split()) < MIN_LENGTH_TO_CHECK:
        return False
    try:
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return similarity >= threshold
    except Exception as e:
        print(f"Similarity check failed: {e}")
        return False

# Recalculate and update label based on new token count
def relabel_chunk(tokens: int) -> str:
    if tokens >= 400:
        return "heavy"
    elif tokens >= 150:
        return "medium"
    else:
        return "light"

# Merge semantically similar chunks
def merge_similar_chunks(chunks: List[Dict]) -> List[Dict]:
    if not chunks:
        return []

    merged_chunks = [chunks[0]]

    for i, curr in enumerate(chunks[1:], 1):
        prev = merged_chunks[-1]
        print(f"Comparing chunk {i} / {len(chunks)}")

        if chunks_are_related(prev["text"], curr["text"]):
            prev["text"] += "\n" + curr["text"]
            prev["tokens"] += curr.get("tokens", 0)
            prev["label"] = relabel_chunk(prev["tokens"])
        else:
            merged_chunks.append(curr)

    print(f"Merging complete. Total merged chunks: {len(merged_chunks)}")
    return merged_chunks