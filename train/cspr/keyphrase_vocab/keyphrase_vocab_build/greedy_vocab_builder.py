import heapq, json, os
from tqdm import tqdm
from argparse import ArgumentParser
from train.cspr.keyphrase_vocab.utils import _load_keyphrase_index


def _apply_grouping(keyphrase_index: dict, grouping: dict) -> dict:
    """Replace every phrase with its canonical and union their doc sets."""
    member_to_canonical = {
        member: canonical
        for canonical, members in grouping.items()
        for member in members
    }
    merged: dict[str, set] = {}
    for phrase, docs in keyphrase_index.items():
        canonical = member_to_canonical.get(phrase, phrase)
        if canonical not in merged:
            merged[canonical] = set()
        merged[canonical].update(docs)
    return {k: list(v) for k, v in merged.items()}


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type = str, help = "Phrase occurrences folder")
    parser.add_argument("--num_phrases", type = int, default = 30000)
    parser.add_argument("--output_file", type = str, default = "vocab_gitig_.json")
    parser.add_argument("--phrase_min_frequency", type = int, default = 20)
    parser.add_argument("--grouping_file", type = str, default = None,
                        help = "JSON from keyphrase grouping: {canonical: [member, ...]}. "
                               "When provided, members are merged under their canonical before running greedy.")
    args = parser.parse_args()

    input_folder = args.input_folder
    num_phrases = args.num_phrases
    output_file = args.output_file
    phrase_min_frequency = args.phrase_min_frequency

    keyphrase_index = _load_keyphrase_index(input_folder)
    keyphrase_index = {k: v for k, v in keyphrase_index.items() if len(v) >= phrase_min_frequency}

    grouping = None
    if args.grouping_file:
        with open(args.grouping_file) as f:
            grouping = json.load(f)
        keyphrase_index = _apply_grouping(keyphrase_index, grouping)
        print(f"After grouping: {len(keyphrase_index)} canonical phrases")

    phrase_vocab = list(sorted(keyphrase_index.keys()))

    sets = [set(keyphrase_index[phrase]) for phrase in phrase_vocab]
    universe = set([])
    for phrase in phrase_vocab:
        universe.update(keyphrase_index[phrase])

    print("Total number of phrases", len(phrase_vocab), f". Will choose {num_phrases} from these")

    selected, covered = greedy_max_coverage_optimized(sets, universe, num_phrases)

    selected_canonicals = [phrase_vocab[i] for i in selected]
    if grouping is not None:
        vocab = [(c, grouping.get(c, [c])) for c in selected_canonicals]
    else:
        vocab = [(c, [c]) for c in selected_canonicals]

    print(f"New phrase vocab covered {len(covered) / len(universe)}% of documents")

    with open(output_file, "w") as f:
        json.dump(vocab, f, indent = 2, ensure_ascii = False)


if __name__ == "__main__":
    main()