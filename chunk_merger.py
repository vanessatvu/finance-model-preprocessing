from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# CONFIGS
SIMILARITY_THRESHOLD = 0.80 # propose to make this threshold higher (0.85/0.9) to reduce chunk length so chunks that are only very semantically similar are combined maybe ?
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

# first pass: merge semantically similar chunks
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

# second pass to rechunk merged chunks semantically
def semantic_rechunk_pass(merged_chunks: List[Dict]) -> List[Dict]:
    embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    semantic_chunker = SemanticChunker(embeddings=embedding_function)

    # set params
    semantic_chunker.chunk_size = 400
    semantic_chunker.chunk_overlap = 50

    final_chunks = []
    for i, chunk in enumerate(merged_chunks):
        text = chunk["text"]
        print(f"Rechunking merged chunk {i} / {len(merged_chunks)} (original tokens: {chunk.get('tokens', 0)})")
        try:
            split_texts = semantic_chunker.split_text(text)
            for split in split_texts:
                final_chunks.append({
                    "text": split,
                    "tokens": len(split.split()),
                    "label": relabel_chunk(len(split.split()))
                })
            print(f"Created {len(split_texts)} subchunks from chunk {i}")
        except Exception as e:
            print(f"Failed to rechunk chunk {i}: {e}")
            final_chunks.append(chunk)

    print(f"Semantic rechunking complete. Final chunk count: {len(final_chunks)}")
    return final_chunks