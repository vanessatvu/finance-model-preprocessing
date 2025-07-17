# utils.py
import json
import re
import unicodedata

def clean_paragraph(para):
    """
    Cleans a paragraph of unwanted characters, unicode artifacts, headers, footers, and links.
    """
    # Normalize fancy unicode characters (e.g. em dashes, smart quotes)
    para = unicodedata.normalize("NFKD", para)

    # Remove visible unicode escape sequences like \u2019 or \u000a
    para = re.sub(r'\\u[\dA-Fa-f]{4}', '', para)

    # Remove raw unicode-like artifacts after decoding
    para = para.encode("ascii", "ignore").decode("ascii")

    # Remove known page headers and boilerplate
    boilerplate_patterns = [
        r'IMF WORKING PAPER.*?PAPER',
        r'International Monetary Fund.*?Fund',
        r'Contents of .*?_final_text_document\.txt.*?TEXT',
        r'--- Page page_.*?\.png ---',
        r'Page page_.*?\.png',
    ]
    for pattern in boilerplate_patterns:
        para = re.sub(pattern, '', para, flags=re.IGNORECASE)

    # Remove raw URLs
    para = re.sub(r'http\S+', '', para)

    # Strip line breaks, slashes, and excess whitespace
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
