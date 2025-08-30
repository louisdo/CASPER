from argparse import ArgumentParser
from transformers import AutoModelForMaskedLM, AutoTokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--intermediate_checkpoint_dir", type = str, required = True)
    parser.add_argument("--source_model_name", type = str, required = True)
    parser.add_argument("--huggingface_path", type = str, required = True, help = "Intermediate model will be saved to this path on Huggingface")

    args = parser.parse_args()

    intermediate_checkpoint_dir = args.intermediate_checkpoint_dir
    source_model_name = args.source_model_name
    huggingface_path = args.huggingface_path

    model = AutoModelForMaskedLM.from_pretrained(intermediate_checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(source_model_name)

    tokenizer.push_to_hub(huggingface_path)
    model.push_to_hub(huggingface_path)


if __name__ == "__main__":
    main()