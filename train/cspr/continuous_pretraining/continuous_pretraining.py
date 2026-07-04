import os, json
from typing import List, Tuple
from argparse import ArgumentParser
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from train.cspr.tokenizer import format_keyphrase, TextProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class KeyphraseWeightedDataCollator(DataCollatorForLanguageModeling):

    def __init__(self, *args, num_original_tokens: int, keyphrase_mlm_probability: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_original_tokens = num_original_tokens
        self.keyphrase_mlm_probability = keyphrase_mlm_probability

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # raise the mask probability on keyphrase-token positions
        is_keyphrase = inputs >= self.num_original_tokens
        probability_matrix[is_keyphrase] = self.keyphrase_mlm_probability

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # only compute loss on masked tokens

        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with a random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # the remaining 10% keep the original token
        return inputs, labels


def init_model_for_continuous_pretraining(
        base_model_name: str,
        keyphrase_vocab: List[Tuple[str, List[str]]],
        max_keyphrase: int = 15000,
        trust_remote_code: bool = False,
        truncation_side: str = "left"):

    model = AutoModelForMaskedLM.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        truncation_side=truncation_side,
        trust_remote_code=trust_remote_code
    )

    # add keyphrase vocabulary
    keyphrase_vocab = keyphrase_vocab[:max_keyphrase]
    new_tokens = [format_keyphrase(canonical) for canonical, _ in keyphrase_vocab]

    num_added = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added}/{len(new_tokens)} keyphrase tokens to the tokenizer")

    # resize model
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def freeze_except_keyphrase_embeddings(model, num_original_tokens: int):
    for p in model.parameters():
        p.requires_grad = False

    def zero_original_rows(grad):
        grad = grad.clone()
        grad[:num_original_tokens] = 0
        return grad

    embeddings = model.get_input_embeddings()
    embeddings.weight.requires_grad = True
    embeddings.weight.register_hook(zero_original_rows)

    decoder = model.get_output_embeddings()
    if decoder is not None and decoder.bias is not None:
        decoder.bias.requires_grad = True
        decoder.bias.register_hook(zero_original_rows)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(
        f"Training {len(embeddings.weight) - num_original_tokens} keyphrase rows; "
        f"{num_trainable}/{num_total} parameters carry gradients "
        f"(original rows are masked to zero in the backward pass)"
    )

    return model


def load_keyphrase_vocab(path: str) -> List[Tuple[str, List[str]]]:
    with open(path) as f:
        return json.load(f)


class S2ORCDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, max_seq_length: int, keyphrase_vocab_path: str,
                 max_keyphrase: int, mode: str = "append",
                 keep_surface: bool = False, num_original_tokens: int = 0,
                 sep_token: str = "[SEP]", sample_size: int = 0, sampling_seed: int = 0,
                 cache_dir: str = None):
        if mode not in ("append", "inplace"):
            raise ValueError(f"Unknown mode '{mode}'. Expected 'append' or 'inplace'.")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.keep_surface = keep_surface
        self.num_original_tokens = num_original_tokens

        self.processor = TextProcessor(
            keyphrase_vocab_path=keyphrase_vocab_path,
            sep_token=sep_token,
            max_keyphrase=max_keyphrase,
        )

        # memory-mapped; rows are not loaded until accessed
        ds = load_dataset(
            "sentence-transformers/s2orc", "title-abstract-pair",
            split="train", cache_dir=cache_dir,
        )
        if sample_size > 0 and sample_size < len(ds):
            ds = ds.shuffle(seed=sampling_seed).select(range(sample_size))
        self.dataset = ds

    def __getitem__(self, idx):
        line = self.dataset[idx]
        text = f"{line.get('title', '')}. {line.get('abstract', '')}".strip()

        if self.mode == "append":
            processed = self.processor.process_each_text(text, add_negatives=False)
            if processed is None:  # no positive keyphrase
                processed = text
        else:  # inplace (returns the original text unchanged when no positive keyphrase)
            processed = self.processor.process_each_text_inplace(text, keep_surface=self.keep_surface)

        return self.tokenizer(
            processed,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.dataset)


def main():
    parser = ArgumentParser(description="Continuous MLM pretraining for keyphrase embeddings")
    parser.add_argument("--base-model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keyphrase-vocab", type=str, required=True,
                        help="Path to keyphrase vocabulary json")
    parser.add_argument("--s2orc-sample-size", type=int, default=0,
                        help="If >0, subsample this many S2ORC docs")
    parser.add_argument("--s2orc-seed", type=int, default=123,
                        help="Seed for the S2ORC subsample")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Where HF datasets stores the S2ORC download/cache "
                        "(defaults to ~/.cache/huggingface/datasets)")
    parser.add_argument("--mode", type=str, default="append", choices=["append", "inplace"],
                        help="Keyphrase processing, matching pretraining_data_processing.py: "
                        "'append' adds keyphrase tokens after [SEP]; 'inplace' rewrites them "
                        "where they occur in the text.")
    parser.add_argument("--keep-surface", action="store_true",
                        help="inplace mode only: keep the surface words and add the "
                        "keyphrase token next to them instead of replacing")
    parser.add_argument("--sep-token", type=str, default="[SEP]",
                        help="Separator between text and appended keyphrases (append mode)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-keyphrase", type=int, default=15000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--truncation-side", type=str, default="left", choices=["left", "right"],
                        help="Side to truncate long docs from. 'left' keeps the end "
                        "(append scheme, keyphrases after [SEP]); 'right' keeps the start "
                        "(inplace scheme, keyphrases spread through the text).")
    parser.add_argument("--mlm-probability", type=float, default=0.3)
    parser.add_argument("--keyphrase-mlm-probability", type=float, default=0.5,
        help="Mask probability applied to keyphrase tokens (id >= num_original_tokens)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="If >0, cap training at this many optimizer steps "
                        "(overrides --num-train-epochs)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-only-keyphrase-embeddings", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0,
                        help="Subprocesses for data loading (0 = load in the main process)")
    args = parser.parse_args()

    # Tokens present before keyphrases are added; rows at or beyond this index are
    # the only trainable embedding rows.
    num_original_tokens = AutoConfig.from_pretrained(
        args.base_model_name, 
        trust_remote_code=args.trust_remote_code).vocab_size

    keyphrase_vocab = load_keyphrase_vocab(args.keyphrase_vocab)
    model, tokenizer = init_model_for_continuous_pretraining(
        args.base_model_name, keyphrase_vocab,
        max_keyphrase=args.max_keyphrase,
        trust_remote_code=args.trust_remote_code,
        truncation_side=args.truncation_side
    )

    if args.train_only_keyphrase_embeddings:
        freeze_except_keyphrase_embeddings(model, num_original_tokens)

    train_dataset = S2ORCDataset(
        tokenizer,
        args.max_seq_length,
        keyphrase_vocab_path=args.keyphrase_vocab,
        max_keyphrase=args.max_keyphrase,
        mode=args.mode,
        keep_surface=args.keep_surface,
        num_original_tokens=num_original_tokens,
        sep_token=args.sep_token,
        sample_size=args.s2orc_sample_size,
        sampling_seed=args.s2orc_seed,
        cache_dir=args.cache_dir,
    )

    data_collator = KeyphraseWeightedDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        num_original_tokens=num_original_tokens,
        keyphrase_mlm_probability=args.keyphrase_mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        seed=args.seed,
        report_to="none",
        save_total_limit=3,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved continuously-pretrained model to {args.output_dir}")


if __name__ == "__main__":
    main()
