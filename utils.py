# utils.py
import json
import re
import unicodedata

def clean_paragraph(para):
    para = unicodedata.normalize("NFKD", para)
    para = re.sub(r'\\u[\dA-Fa-f]{4}', '', para)
    para = para.encode("ascii", "ignore").decode("ascii")

    boilerplate_patterns = [
        r'IMF WORKING PAPER.*?PAPER',
        r'International Monetary Fund.*?Fund',
        r'Contents of .*?_final_text_document\.txt.*?TEXT',
        r'--- Page page_.*?\.png ---',
        r'Page page_.*?\.png',
    ]
    for pattern in boilerplate_patterns:
        para = re.sub(pattern, '', para, flags=re.IGNORECASE)

    para = re.sub(r'http\S+', '', para)

    para = para.replace('\\n', ' ').replace('\n', ' ').replace('\\', '')
    para = re.sub(r'\s+', ' ', para).strip()

    return para


def save_to_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")

def save_to_txt(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(entry["text"] + "\n\n")
