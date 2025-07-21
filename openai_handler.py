import openai
import time
from transformers import AutoTokenizer
import os

client = openai.OpenAI(api_key="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")  # Replace with your real key or use an environment variable

mos.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("gpt4")  # or use your model's tokenizer
MAX_TOKENS = 8000  # safe margin under 8192

def truncate_prompt(prompt: str, max_tokens: int = MAX_TOKENS):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)

def merge_similar_chunks(chunks):
    merged_chunks = []
    if not chunks:
        return merged_chunks

    current_merged = chunks[0]["text"]

    for i in range(1, len(chunks)):
        prev = current_merged
        curr = chunks[i]["text"]

        raw_prompt = f"""
You are given two paragraphs from a financial document.

Paragraph 1:
{prev}

Paragraph 2:
{curr}

Are these two paragraphs discussing the same topic or are they very closely related in content and context?
Reply with "yes" if they should be merged, or "no" if they are different and should stay separate.
"""
        prompt = truncate_prompt(raw_prompt)

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content.strip().lower()
            if "yes" in answer:
                current_merged += "\n\n" + curr
            else:
                merged_chunks.append({"text": current_merged})
                current_merged = curr

            time.sleep(1.5)

        except Exception as e:
            print("OpenAI error:", e)
            merged_chunks.append({"text": current_merged})
            current_merged = curr

    merged_chunks.append({"text": current_merged})
    return merged_chunks