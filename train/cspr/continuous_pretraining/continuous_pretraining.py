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
from train.cspr.tokenizer import format_keyphrase

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class KeyphraseWeightedDataCollator(DataCollatorForLanguageModeling):

    def __init__(self, *args, num_original_tokens: int, keyphrase_mlm_probability: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_original_tokens = num_original_tokens
        self.keyphrase_mlm_probability = keyphrase_mlm_probability

    def __call__(self, examples):
        has_negatives = len(examples) > 0 and "negative_keyphrase_ids" in examples[0]
        if has_negatives:
            negative_ids = [example.pop("negative_keyphrase_ids") for example in examples]

        batch = super().__call__(examples)

        if has_negatives:
            max_neg = max((len(negs) for negs in negative_ids), default=0)
            neg_tensor = torch.zeros((len(examples), max_neg), dtype=torch.long)
            neg_mask = torch.zeros((len(examples), max_neg), dtype=torch.bool)
            for i, negs in enumerate(negative_ids):
                if negs:
                    neg_tensor[i, : len(negs)] = torch.tensor(negs, dtype=torch.long)
                    neg_mask[i, : len(negs)] = True
            batch["negative_keyphrase_ids"] = neg_tensor
            batch["negative_keyphrase_mask"] = neg_mask

        return batch

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


class UnlikelihoodTrainer(Trainer):
    def __init__(self, *args, num_original_tokens: int, ul_alpha: float = 1.0, ul_eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_original_tokens = num_original_tokens
        self.ul_alpha = ul_alpha
        self.ul_eps = ul_eps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        negative_ids = inputs.pop("negative_keyphrase_ids", None)
        negative_mask = inputs.pop("negative_keyphrase_mask", None)
        labels = inputs["labels"]

        outputs = model(**inputs)  # labels present -> outputs.loss is the MLM cross-entropy
        loss = outputs.loss

        # masked keyphrase slots: a loss is computed there (label != -100) and the
        # gold token is a keyphrase (id >= num_original_tokens).
        kp_pos = (labels != -100) & (labels >= self.num_original_tokens)
        if (
            negative_ids is not None
            and negative_ids.numel() > 0
            and kp_pos.any()
        ):
            B, T = labels.shape
            # gather only the masked-keyphrase rows -> (M, V) instead of (B, T, V)
            sel_logits = outputs.logits[kp_pos]  # (M, V)
            batch_idx = torch.arange(B, device=labels.device)[:, None].expand(B, T)[kp_pos]  # (M,)
            neg_for_pos = negative_ids[batch_idx]  # (M, N)
            mask_for_pos = negative_mask[batch_idx]  # (M, N)

            probs = sel_logits.float().softmax(dim=-1)  # upcast for stable log(1 - p)
            neg_probs = probs.gather(1, neg_for_pos)  # (M, N); padded ids gather id 0, masked out below
            ul = -torch.log((1.0 - neg_probs).clamp(min=self.ul_eps))  # (M, N)

            denom = mask_for_pos.sum().clamp(min=1)
            ul_loss = (ul * mask_for_pos).sum() / denom
            loss = loss + self.ul_alpha * ul_loss

        return (loss, outputs) if return_outputs else loss



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


NEG_SEP = "||NEG||"


def build_dataset(
    train_file: str,
    tokenizer,
    text_column: str,
    max_seq_length: int,
    num_original_tokens: int,
    parse_negatives: bool = False,
    max_samples: int = 0,
):
    split = f"train[:{max_samples}]" if max_samples > 0 else "train"
    if train_file.endswith(".txt"):
        dataset = load_dataset("text", data_files=train_file, split=split)
        text_column = "text"
    else:
        dataset = load_dataset("json", data_files=train_file, split=split)

    def tokenize(batch):
        bodies, neg_strings = [], []
        for line in batch[text_column]:
            body, _, negs = line.partition(NEG_SEP)
            bodies.append(body)
            neg_strings.append(negs)

        enc = tokenizer(
            bodies,
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

        if parse_negatives:
            neg_encoded = tokenizer(neg_strings, add_special_tokens=False)["input_ids"]
            enc["negative_keyphrase_ids"] = [
                [i for i in ids if i >= num_original_tokens] for ids in neg_encoded
            ]

        return enc

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def main():
    parser = ArgumentParser(description="Continuous MLM pretraining for keyphrase embeddings")
    parser.add_argument("--base-model-name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keyphrase-vocab", type=str, required=True, 
                        help="Path to keyphrase vocabulary json")
    parser.add_argument("--train-file", type=str, required=True, 
                        help=".txt (one doc per line) or .jsonl with a text field")
    parser.add_argument("--text-column", type=str, default="text")
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
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="If >0, load only the first N rows of the corpus (for smoke testing)",)
    parser.add_argument("--unlikelihood", action="store_true",
                        help="Add an unlikelihood term on the per-document negative keyphrases "
                        "(parsed from the '||NEG||' marker). Off = plain MLM continuous pretraining.")
    parser.add_argument("--ul-alpha", type=float, default=1.0,
                        help="Weight of the unlikelihood term (only used with --unlikelihood)")
    parser.add_argument("--ul-eps", type=float, default=1e-6,
                        help="Clamp floor for log(1 - p) in the unlikelihood term")
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

    train_dataset = build_dataset(
        args.train_file,
        tokenizer,
        args.text_column,
        args.max_seq_length,
        num_original_tokens,
        parse_negatives=args.unlikelihood,
        max_samples=args.max_samples,
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

    if args.unlikelihood:
        trainer = UnlikelihoodTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            num_original_tokens=num_original_tokens,
            ul_alpha=args.ul_alpha,
            ul_eps=args.ul_eps,
        )
    else:
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
