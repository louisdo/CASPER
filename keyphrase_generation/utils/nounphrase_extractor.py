
import nltk, string, re
from collections import Counter
from functools import lru_cache



CANDEXT = {
}

class CandidateExtractorRegExpNLTK:
    """
    CandidateExtractorRegExpNLTK class extracts candidate phrases from a given text using regular expressions
    and Natural Language Toolkit (NLTK) tools.

    Args:
    - length_range (list, optional): A list specifying the minimum and maximum length of candidate phrases.
      Default is [2, 5].
    - cand_counter (Counter, optional): A Counter object to track the occurrences of candidate phrases.
      Default is None.

    Attributes:
    - np_parser (nltk.RegexpParser): A RegexpParser object for parsing noun phrases based on the specified grammar.
    - length_range (list): A list specifying the minimum and maximum length of candidate phrases.
    - cand_counter (Counter): A Counter object to track the occurrences of candidate phrases.

    Methods:
    - word_check(word): Checks if a word is valid based on a set of conditions.
    - get_best_subphrase(tokens): Retrieves the best subphrase from a list of tokens based on the candidate counter.
    - get_phrase_from_subtree(subtree): Extracts a candidate phrase from a given subtree.
    - __call__(text): Extracts candidate phrases from the input text.

    Examples:
    >>> extractor = CandidateExtractorRegExpNLTK(length_range=[2, 5])
    >>> text = "This is a sample text with candidate phrases."
    >>> candidates = extractor(text)
    """

    def __init__(self, length_range=[2, 5], cand_counter=None):
        GRAMMAR_EN = r"""
        TT: {(<NN.*|JJ.*>+<NN.*|CD>)|<NN.*>}
        """
        self.np_parser = nltk.RegexpParser(GRAMMAR_EN)
        self.length_range = length_range
        self.cand_counter = cand_counter

    @staticmethod
    @lru_cache(maxsize=10000)
    def word_check(word):
        if word in string.punctuation: return False
        return True
        if len(word) > 1 and re.search(r"^([a-zA-Z]+-)*[a-zA-Z]+((\s|-)[0-9]+){0,1}$", word): # r"^([a-z]+-)*[a-z]+(\s[0-9]+){0,1}$"
            return True
        return False

    def get_best_subphrase(self, tokens):
        print("using get best subphrase")
        all_subphrases = []
        for length in range(2, self.length_range[1] + 1):
            all_subphrases.extend(
                [
                    " ".join(tokens[index : index + length])
                    for index in range(len(tokens) - length + 1)
                ]
            )
        all_subphrases = [p for p in all_subphrases if self.cand_counter[p] > 0]

        if len(all_subphrases) == 0:
            return []
        res = max(all_subphrases, key=lambda x: self.cand_counter[x])

        return res.split(" ")

    def get_phrase_from_subtree(self, subtree):
        exp = []
        flag = True
        for l in subtree.leaves():
            w = str(l[0])
            if not self.word_check(w):
                flag = False
                break
            exp.append(w)
        if len(exp) > self.length_range[1]:
            if not self.cand_counter:
                flag = False
            else:
                exp = self.get_best_subphrase(exp)
                if len(exp) == 0:
                    flag = False
            exp = exp[-self.length_range[1] :]
        phrase_length = len(exp)
        exp = " ".join(exp)
        if flag:
            return exp, phrase_length
        return None, 0

    def __call__(self, text):
        tokenized = nltk.word_tokenize(text)
        # text_token_counter = Counter(tokenized)
        tags = nltk.pos_tag(tokenized)

        tree = self.np_parser.parse(tags)

        cands = []
        cands_counter = Counter()
        for st in list(tree.subtrees(filter=lambda x: x.label().endswith("TT"))):
            if len(st.leaves()) < self.length_range[0]:
                continue

            cand, cand_length = self.get_phrase_from_subtree(st)
            if cand is None:
                continue
            cands_counter[cand] += 1
            if cands_counter[cand] == 1:
                cands.append(cand)
        return [candidate.lower() for candidate in list(cands)]


def init_candext(length_range=[2, 5]):
    name = f"candext_{length_range[0]}_{length_range[1]}"
    if not CANDEXT.get(name):
        CANDEXT[name] = CandidateExtractorRegExpNLTK(length_range)

def nounphrase_extraction_as_keyphrase_generation(doc, length_range):
    name = f"candext_{length_range[0]}_{length_range[1]}"
    init_candext(length_range)

    return {
        "present": [[item, 1] for item in CANDEXT[name](doc)],
        "absent": []
    }
