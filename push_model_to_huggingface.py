from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type = str, required = True)
    parser.add_argument("--model_name_on_hf", type = str, required = True)

    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    model_name_on_hf = args.model_name_on_hf

    model = AutoModel.from_pretrained("/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_46/debug/checkpoint/model")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_46/debug/checkpoint/model")