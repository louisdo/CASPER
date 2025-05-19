# python greedy_vocab_builder.py --input_folder /scratch/lamdo/msmarco_phrase_vocab/ --num_phrases 30000 --output_file phrase_vocab_msmarco_gitig_.json
# python greedy_vocab_builder.py --input_folder /scratch/lamdo/s2orc_word_vocab/ --num_phrases 30000 --output_file word_vocab_s2orc_gitig_.json --apply_word_check 1
import json, os, heapq, string, re
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import lru_cache
from english_words import get_english_words_set

TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
WEB2LOWERSET = get_english_words_set(['web2'], lower=True)


def greedy_max_coverage_optimized(sets, universe, k):
    # Precompute initial coverage for each set
    uncovered = set(universe)
    set_indices = list(range(len(sets)))
    heap = []
    for i in set_indices:
        # Use negative for max-heap
        heapq.heappush(heap, (-len(sets[i] & uncovered), i))
    
    selected = []
    covered = set()
    used = set()
    
    for _ in tqdm(range(k), desc = "Creating vocabulary"):
        while heap:
            neg_gain, idx = heapq.heappop(heap)
            # Recompute actual gain due to dynamic uncovered set
            gain = len(sets[idx] - covered)
            if gain == -neg_gain and idx not in used:
                selected.append(idx)
                covered |= sets[idx]
                used.add(idx)
                break
            elif gain > 0 and idx not in used:
                # Push updated gain back into heap
                heapq.heappush(heap, (-gain, idx))
    
    return selected, covered


def greedy_max_coverage(sets, universe, k):
    uncovered = set(universe)
    selected_sets = []
    
    for _ in tqdm(range(k), desc = "Running greedy algorithm"):
        best_set = None
        best_set_index = None
        max_cover = 0
        
        # Find the set covering most uncovered elements
        for i, s in enumerate(sets):
            current_cover = len(s & uncovered)
            if current_cover > max_cover:
                max_cover = current_cover
                best_set = s
                best_set_index = i
                
        # No remaining coverage possible
        if best_set is None:
            break  
            
        selected_sets.append(best_set_index)
        uncovered -= best_set
    
    covered = len(universe) - len(uncovered)
    return selected_sets, covered

@lru_cache(maxsize=100000)
def check_word_token_length(word):
    return len(TOKENIZER.tokenize(word)) >= 3


def word_check(word):
    if word in string.punctuation: 
        return False
    
    if word not in WEB2LOWERSET: return False

    if len(word) > 1 and re.search(r"^([a-zA-Z]+-)*[a-zA-Z]+((\s|-)[0-9]+){0,1}$", word): # r"^([a-z]+-)*[a-z]+(\s[0-9]+){0,1}$"
        return True
        
    return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, help = "Phrase occurrences folder")
    parser.add_argument("--num_phrases", type = int, default = 30000)
    parser.add_argument("--output_file", type = str, default = "vocab_gitig_.json")
    parser.add_argument("--phrase_min_frequency", type = int, default = 20)
    parser.add_argument("--apply_word_check", type = int, choices = [0, 1], default = 0, 
                        help = "This should be 1 only if we are adding words into the model")

    args = parser.parse_args()

    input_folder = args.input_folder
    num_phrases = args.num_phrases
    output_file = args.output_file
    phrase_min_frequency = args.phrase_min_frequency
    apply_word_check = args.apply_word_check

    bert_vocab = TOKENIZER.vocab

    input_files = os.listdir(input_folder)
    input_files = [os.path.join(input_folder, file) for file in input_files]


    phrase_occurrences = {}
    for file in input_files:
        print("Loading file", file)
        with open(file) as f:
            temp = json.load(f)

            for k in tqdm(temp, total = len(temp)):
                if k not in phrase_occurrences: phrase_occurrences[k] = []
                phrase_occurrences[k].extend(temp[k])
    phrase_occurrences = {k:v for k,v in phrase_occurrences.items() if len(v) >= phrase_min_frequency and k not in bert_vocab}
    if apply_word_check:
        phrase_occurrences = {k:v for k,v in tqdm(phrase_occurrences.items(), desc = "Running word check", total = len(phrase_occurrences)) if word_check(k)}

    phrase_vocab = list(sorted(phrase_occurrences.keys()))

    sets = [set(phrase_occurrences[phrase]) for phrase in phrase_vocab]
    universe = set([])
    for phrase in phrase_vocab:
        universe.update(phrase_occurrences[phrase])

    print("Total number of phrases", len(phrase_vocab), f". Will choose {num_phrases} from these")

    selected, covered = greedy_max_coverage_optimized(sets, universe, num_phrases)
    test = [phrase_vocab[i] for i in selected]

    print(f"New phrase vocab covered {len(covered) / len(universe)}% of documents")

    with open(output_file, "w") as f:
        json.dump(test, f, indent = 4)

if __name__ == "__main__":
    main()