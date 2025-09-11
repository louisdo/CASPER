import gzip, json, nltk, re, os
from tqdm import tqdm
from nltk.tokenize import PunktTokenizer

def slightly_process_fulltext(fulltext):
    if not fulltext: return ""
    fulltext = fulltext.replace("et al.", "et al ") # to enable better sentence tokenization
    return fulltext

def extract_data_from_paper(paper):
    content = paper.get("content", {})
    fulltext = slightly_process_fulltext(content.get("text", ""))
    annotations = content.get("annotations", {})

    bibentry = annotations.get("bibentry")
    bibentry = json.loads(bibentry) if bibentry else []

    title_position = annotations.get("title")
    title_position = json.loads(title_position) if title_position else None
    title_position = title_position[0] if title_position else None
    abstract_position = annotations.get("abstract")
    abstract_position = json.loads(abstract_position) if abstract_position else None
    abstract_position = abstract_position[0] if abstract_position else None

    bibentry = [be for be in bibentry if be.get("attributes", {}).get("matched_paper_id")]

    