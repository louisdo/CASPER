import gzip, json, nltk, re, os
from tqdm import tqdm
from nltk.tokenize import PunktTokenizer

SENT_TOKENIZER = PunktTokenizer("english")


def maybe_create_folder(path):
    """
    Creates a directory if it does not exist.

    Parameters:
    path (str): The path of the directory to be created.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created successfully!")
        else:
            print(f"Directory '{path}' already exists.")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")

def group_bibref(bibref):
    if not bibref: return []
    res = []
    current_group = [bibref[0]]

    for i in range(1, len(bibref)):
        current_start = bibref[i].get("start")
        prev_end = bibref[i - 1].get("end")

        if not isinstance(current_start, int) or not isinstance(prev_end, int): continue

        if not current_start or not prev_end: continue

        if current_start - prev_end < 5:
            current_group.append(bibref[i])
        else:
            res.append(current_group)
            current_group = [bibref[i]]

    return res


def binary_search_span(arr, target, text_length):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        mid_val = arr[mid]
        upper_val = arr[mid + 1] if mid + 1 < len(arr) else text_length
        
        if mid_val <= target < upper_val:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def span_sentence_tokenize(text):
    return list(SENT_TOKENIZER.span_tokenize(text))


def slightly_process_fulltext(fulltext):
    if not fulltext: return ""
    fulltext = fulltext.replace("et al.", "et al ")
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

    # filter the bibentry, keep only those that have "matched_paper_id"
    bibentry = [be for be in bibentry if be.get("attributes", {}).get("matched_paper_id")]
    if not bibentry: return []

    ref_id_2_matched_paper_id = {be.get("attributes", {}).get("id"): be.get("attributes", {}).get("matched_paper_id") for be in bibentry}

    if not fulltext: return []

    ref_id_2_matched_paper_id = {}
    for be in bibentry:
        be_attributes = be.get("attributes", {})
        ref_id = be_attributes.get("id")
        matched_paper_id = be_attributes.get("matched_paper_id")

        if not ref_id or not matched_paper_id: continue

        ref_id_2_matched_paper_id[ref_id] = matched_paper_id

    if not ref_id_2_matched_paper_id: return []

    bibref = annotations.get("bibref")
    bibref = json.loads(bibref) if bibref else []
    grouped_bibref = group_bibref(bibref)

    # process full text, remove the bibref
    for br in bibref:
        br_start = br["start"]
        br_end = br["end"]

        if not isinstance(br_start, int) or not isinstance(br_end, int): continue

        fulltext = fulltext[:br_start] + " " * (br_end - br_start) + fulltext[br_end:]

    # tokenize into sentences to extract citation sentences
    sentences_spans = span_sentence_tokenize(fulltext)
    sentences_spans_start = [item[0] for item in sentences_spans]

    sentence_2_ref = [[] for _ in range(len(sentences_spans))]
    for group in grouped_bibref:
        start = group[0]["start"]
        sentence_index = binary_search_span(sentences_spans_start, start, len(fulltext))
        sentence_2_ref[sentence_index].append(group)

    citation_contexts_data = []
    for i in range(len(sentence_2_ref)):
        if len(sentence_2_ref[i]) != 1: continue

        sentence_text = fulltext[sentences_spans[i][0]: sentences_spans[i][1]]
        sentence_text = re.sub(r'\s+', ' ', sentence_text)

        if sentence_text.count(" ") < 8: continue

        cited_ref_ids = [item.get("attributes", {}).get("ref_id") for item in sentence_2_ref[i][0]]
        cited_corpus_ids = [ref_id_2_matched_paper_id.get(cri) for cri in cited_ref_ids if cri]
        cited_corpus_ids = [cci for cci in cited_corpus_ids if cci]

        if cited_corpus_ids: citation_contexts_data.append([sentence_text, cited_corpus_ids])

    
    co_citation_data = []
    for group in grouped_bibref:
        group_ref_ids = [item.get("attributes", {}).get("ref_id") for item in group]
        group_ref_ids = [gri for gri in group_ref_ids if gri]
        group_corpus_ids = [ref_id_2_matched_paper_id.get(gri) for gri in group_ref_ids]
        group_corpus_ids = [gci for gci in group_corpus_ids if gci]
        
        if len(group_corpus_ids) >= 2: co_citation_data.append(group_corpus_ids)

    try:
        title = fulltext[title_position["start"]: title_position["end"]] if title_position else ""
        abstract = fulltext[abstract_position["start"]: abstract_position["end"]] if abstract_position else ""
    except TypeError: 
        title, abstract = "", ""
    

    return {
        "citation_context": citation_contexts_data,
        "title": [title, abstract],
        "co_citation": co_citation_data,
        "all_references": [be.get("attributes", {}).get("matched_paper_id") for be in bibentry]
    }