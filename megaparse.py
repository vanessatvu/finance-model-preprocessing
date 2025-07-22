from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

# Load embedding model (fast + accurate)
model = SentenceTransformer("all-MiniLM-L6-v2")

# CONFIGS
SIMILARITY_THRESHOLD = 0.65  # ↓ Lower = more merging
MIN_LENGTH_TO_CHECK = 10     # Skip comparing junk like "93..."

def chunks_are_related(text1: str, text2: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    # Skip merging if one chunk is extremely short (e.g., chart garbage)
    if len(text1.strip().split()) < MIN_LENGTH_TO_CHECK or len(text2.strip().split()) < MIN_LENGTH_TO_CHECK:
        return False

    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return similarity >= threshold

def relabel_chunk(tokens: int) -> str:
    if tokens >= 400:
        return "heavy"
    elif tokens >= 150:
        return "medium"
    else:
        return "light"

def merge_similar_chunks(chunks: List[Dict]) -> List[Dict]:
    if not chunks:
        return []

    merged_chunks = [chunks[0]]

    for curr in chunks[1:]:
        prev = merged_chunks[-1]

        if chunks_are_related(prev["text"], curr["text"]):
            prev["text"] += "\n" + curr["text"]
            prev["tokens"] += curr.get("tokens", 0)
            prev["label"] = relabel_chunk(prev["tokens"])
        else:
            merged_chunks.append(curr)

    return merged_chunks