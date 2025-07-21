from transformers import AutoTokenizer
from config import MODEL_NAME, TOKEN_THRESHOLDS, MAX_CHUNK_TOKENS
from utils import clean_paragraph

def poor_graphical_text(text):
    if "====" in text or "----" in text:  # lines often seen in tables/ASCII
        return True
    if any(symbol in text for symbol in ['│', '█', '╚', '═']):
        return True
    if sum(c.isdigit() for c in text) > 100:
        return True
    return False

def load_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)

def classify_paragraph(token_count, thresholds=TOKEN_THRESHOLDS):
    if token_count < thresholds["light"]:
        return "light"
    elif token_count <= thresholds["medium"]:
        return "medium"
    else:
        return "heavy"

def chunk_and_classify_text(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split and clean each paragraph
    raw_paragraphs = raw_text.split("\n\n")
    paragraphs = [clean_paragraph(p) for p in raw_paragraphs
              if clean_paragraph(p) and not poor_graphical_text(clean_paragraph(p))]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        tokens = tokenizer.encode(para, add_special_tokens=False)
        token_count = len(tokens)
        label = classify_paragraph(token_count)

        print(f"{label.upper()} | Tokens: {token_count} | {para[:60]}...")

        if current_tokens + token_count > MAX_CHUNK_TOKENS:
            full_chunk = " ".join(current_chunk)
            chunks.append({
                "text": full_chunk,
                "tokens": current_tokens,
                "label": classify_paragraph(current_tokens)
            })
            current_chunk = [para]
            current_tokens = token_count
        else:
            current_chunk.append(para)
            current_tokens += token_count

    if current_chunk:
        full_chunk = " ".join(current_chunk)
        chunks.append({
            "text": full_chunk,
            "tokens": current_tokens,
            "label": classify_paragraph(current_tokens)
        })

    return chunks


