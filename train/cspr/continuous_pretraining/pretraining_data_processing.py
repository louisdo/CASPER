import json, os
from typing import Iterator
from argparse import ArgumentParser
from flashtext import KeywordProcessor
from collections import Counter
from tqdm import tqdm

from train.cspr.tokenizer import format_keyphrase, TextProcessor


def load_corpus(path: str) -> Iterator[str]:
    with open(path) as f:
        for text in f:
            yield text


def main():
    parser = ArgumentParser()
    parser.add_argument("--keyphrase-vocab", type = str, required = True, help = "Path to keyphrase vocabulary")
    parser.add_argument("--s2orc-data", type = str, required = True, help = "Path to S2ORC data")
    parser.add_argument("--sep-token", type = str, default = "[SEP]", help = "Separation token to separate between main text and keyphrases")
    parser.add_argument("--max_keyphrase", type = int, required = True, help = "Number of keyphrase from the keyphrase vocabulary to use")
    parser.add_argument("--add-negatives", action = "store_true", help = "Append negative keyphrases after a '||NEG||' marker (for unlikelihood training)")
    parser.add_argument("--mode", type = str, default = "append", choices = ["append", "inplace"],
                        help = "'append' adds keyphrases after the [SEP]; 'inplace' rewrites keyphrases where they occur in the text")
    parser.add_argument("--keep-surface", action = "store_true",
                        help = "inplace mode only: keep the surface words and add the keyphrase token next to them instead of replacing")
    parser.add_argument("--output", type = str, required = True)
    

    args = parser.parse_args()

    processor = TextProcessor(
        keyphrase_vocab_path = args.keyphrase_vocab,
        sep_token = args.sep_token,
        max_keyphrase = args.max_keyphrase
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok = True)

    with open(args.output, "w") as fout:
        for text in tqdm(load_corpus(args.s2orc_data), desc = "Processing texts"):
            stripped = text.strip()
            if args.mode == "append":
                text_processed = processor.process_each_text(stripped, add_negatives = args.add_negatives)
                if text_processed is None:  # no positive keyphrase
                    continue
            else:  # inplace
                text_processed = processor.process_each_text_inplace(stripped, keep_surface = args.keep_surface)
                if text_processed == stripped:  # no positive keyphrase -> unchanged
                    continue
            fout.write(text_processed + "\n")

if __name__ == "__main__":
    main()