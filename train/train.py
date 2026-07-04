from argparse import ArgumentParser
from datetime import datetime
from typing import Iterator, Dict, List

import torch, os, json, yaml
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

from train.cspr.model import CSpRTrain, CSpRCombinedTrain
from train.cspr.tokenizer import CSpRTokenizer
from train.cspr.losses import MatryoshkaContrastiveLoss, FLOPS, FLOPS


def read_triples(path: str, max_samples = 0) -> Iterator[Dict[str, str]]:
    # if max_samples = 0, use everything
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i == max_samples: break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            yield {"query": parts[0], "pos": parts[1], "neg": parts[2]}


def build_dataset(path: str, max_samples: int = 0) -> Dataset:
    dataset = Dataset.from_generator(lambda: read_triples(path, max_samples))
    return dataset


class TripletCollator:

    def __init__(self, tokenizer, q_max_length: int, d_max_length: int):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def _tok(self, texts: List[str], max_length: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return dict(enc)

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [e["query"] for e in examples]
        pos_documents = [e["pos"] for e in examples]
        neg_documents = [e["neg"] for e in examples]
        return {
            "q_kwargs": self._tok(queries, self.q_max_length),
            "p_kwargs": self._tok(pos_documents, self.d_max_length),
            "n_kwargs": self._tok(neg_documents, self.d_max_length),
        }


def _build_score_dict(
    pos_outputs: Dict[str, torch.Tensor], neg_outputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    out_d: Dict[str, torch.Tensor] = {}
    for prefix, outputs in (("pos", pos_outputs), ("neg", neg_outputs)):
        for key, value in outputs.items():
            if key == "score" or key.startswith("score_"):
                out_d[f"{prefix}_{key}"] = value

    if "pos_score" not in out_d:
        raise RuntimeError(
            "Model returned no 'score' tensor; expected 'score' and 'score_*' keys. "
            "Check that both q_kwargs and d_kwargs were passed."
        )
    return out_d


def _flops_reg(flops_fn: FLOPS, rep_dicts: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    # FLOPS over a partitioned representation. Each rep_dict is {partition: (bs, dim)};
    # several dicts (e.g. positive + negative docs) are pooled along the batch axis.
    # FLOPS sums over feature dims, so summing per-partition equals FLOPS on the
    # concatenated token+phrase vector.
    total = None
    for partition in rep_dicts[0]:
        rep = torch.cat([rd[partition] for rd in rep_dicts], dim=0).float()
        val = flops_fn(rep)
        total = val if total is None else total + val
    return total


class CSpRTrainer(Trainer):
    loss_fn = MatryoshkaContrastiveLoss()
    flops_fn = FLOPS()
    lambda_q = 0.0   # FLOPS weight on query reps (target, after warmup)
    lambda_d = 0.0   # FLOPS weight on document reps (target, after warmup)
    # quadratic warmup for the FLOPS weights: effective lambda = target * (t/T)^2
    # for t < T, then target. 0 disables warmup (full lambda from step 0).
    flops_warmup_steps = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # running sum + count of each loss component over the current logging window
        self._loss_components: Dict[str, float] = {}
        self._loss_component_steps = 0

    def _accumulate_components(self, components: Dict[str, torch.Tensor]):
        for name, value in components.items():
            self._loss_components[name] = self._loss_components.get(name, 0.0) + value.item()
        self._loss_component_steps += 1

    def log(self, logs: Dict[str, float], *args, **kwargs):
        if self._loss_component_steps > 0:
            for name, total in self._loss_components.items():
                logs[name] = round(total / self._loss_component_steps, 4)
            self._loss_components = {}
            self._loss_component_steps = 0
        if getattr(self.loss_fn, "learnable_temperature", False):
            logs["temperature"] = round(self.loss_fn.current_temperature(), 4)
        if self.flops_warmup_steps > 0:
            logs["flops_warmup"] = round(self._flops_warmup_factor(), 4)
        if hasattr(self.model, "current_lambdas"):
            for name, value in self.model.current_lambdas().items():
                logs[f"lambda_{name}"] = round(value, 4)
        return super().log(logs, *args, **kwargs)

    def _flops_warmup_factor(self) -> float:
        # quadratic warm up for FLOPS
        T = self.flops_warmup_steps
        if T <= 0:
            return 1.0
        t = self.state.global_step
        return min(1.0, t / T) ** 2

    def get_decay_parameter_names(self, model):
        decay = super().get_decay_parameter_names(model)
        return [name for name in decay
                if "logit_scale" not in name and "lambda_raw" not in name]

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.sparse_model.transformer.model.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors
        )
        if getattr(self, "processing_class", None) is not None:
            self.processing_class.save_pretrained(output_dir)
        if hasattr(self.model, "current_lambdas"):
            with open(os.path.join(output_dir, "combiner_lambdas.json"), "w") as f:
                json.dump(self.model.current_lambdas(), f, indent=2)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pos_outputs = model(
            q_kwargs=inputs["q_kwargs"],
            d_kwargs=inputs["p_kwargs"],
            score_batch=True,
        )
        
        neg_outputs = model(
            q_rep=pos_outputs["q_rep"],
            d_kwargs=inputs["n_kwargs"],
        )

        out_d = _build_score_dict(pos_outputs, neg_outputs)

        # per-field contrastive losses, logged separately (see log())
        components: Dict[str, torch.Tensor] = {
            f: self.loss_fn.helper(out_d, f, use_hardneg=self.loss_fn.use_hardneg_for(f))
            for f in self.loss_fn.field_names
        }
        loss = sum(components.values())

        # FLOPS regularization on the sparse activations (queries are identical in
        # both passes; documents = positives + negatives). The weights follow a
        # quadratic warmup so the model learns to match before being pushed sparse.
        warmup = self._flops_warmup_factor()
        reg_q = _flops_reg(self.flops_fn, [pos_outputs["q_rep"]])
        reg_d = _flops_reg(self.flops_fn, [pos_outputs["d_rep"], neg_outputs["d_rep"]])
        loss = loss + warmup * self.lambda_q * reg_q + warmup * self.lambda_d * reg_d

        components["reg_q"] = warmup * self.lambda_q * reg_q
        components["reg_d"] = warmup * self.lambda_d * reg_d

        # L0 (mean nonzeros per vector) of each rep partition, so sparsity shows up
        # in the same log line. docs = positives + negatives. Diagnostic only.
        with torch.no_grad():
            for side, rep in (("q", pos_outputs["q_rep"]), ("d", pos_outputs["d_rep"])):
                for part, vec in rep.items():
                    components[f"l0_{side}_{part}"] = (vec > 0).float().sum(dim=1).mean()
            for part, vec in neg_outputs["d_rep"].items():
                components[f"l0_d_{part}"] = (
                    components[f"l0_d_{part}"]
                    + (vec > 0).float().sum(dim=1).mean()
                ) / 2  # average positive- and negative-doc L0

        self._accumulate_components(components)

        if return_outputs:
            return loss, out_d
        return loss



# training classes selectable from config via `train_class`
TRAIN_CLASSES = {
    "CSpRTrain": CSpRTrain,
    "CSpRCombinedTrain": CSpRCombinedTrain,
}


def build_vocab_partitions(tokenizer, original_vocab_size: int) -> Dict[str, List[int]]:
    return {
        "token": list(range(original_vocab_size)),
        "phrase": list(range(original_vocab_size, len(tokenizer))),
    }


def main():
    parser = ArgumentParser(description="CSpR training with in-batch negative log-likelihood")
    parser.add_argument("--config", type=str, default="train/conf/cspr.yaml",
                        help="Path to the YAML training config")
    cli = parser.parse_args()
    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    _tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])
    tok_cfg = cfg["tokenizer"]
    tokenizer = CSpRTokenizer(_tokenizer,
                              keyphrase_vocab_path = tok_cfg["keyphrase_vocab_path"],
                              original_bert_vocab_size = cfg["original_vocab_size"],
                              num_keyphrase_slots = tok_cfg["num_keyphrase_slots"],
                              mask_only = tok_cfg["mask_only"],
                              mode = tok_cfg.get("mode", "append"),
                              keep_surface = tok_cfg.get("keep_surface", False),
                              use_default = tok_cfg.get("use_default", False))

    vocab_partitions = build_vocab_partitions(tokenizer, cfg["original_vocab_size"])

    train_class_name = cfg.get("train_class", "CSpRTrain")
    if train_class_name not in TRAIN_CLASSES:
        raise ValueError(
            f"Unknown train_class '{train_class_name}'. Available: {list(TRAIN_CLASSES)}"
        )
    train_class = TRAIN_CLASSES[train_class_name]

    model_kwargs = dict(
        model_type_or_dir=cfg["model_dir"],
        vocab_partitions=vocab_partitions,
        bf16=cfg["bf16"],
        pooling_method=cfg["pooling_method"],
        pooling_segments=cfg.get("pooling_segments"),
        sep_token_id=_tokenizer.sep_token_id,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if train_class is CSpRCombinedTrain:
        model_kwargs["init_lambda"] = cfg.get("init_lambda", 0.25)

    model = train_class(**model_kwargs)

    train_dataset = build_dataset(cfg["train_file"], cfg.get("max_samples", 0))
    data_collator = TripletCollator(tokenizer, cfg["q_max_length"], cfg["d_max_length"])

    output_dir = os.path.join(cfg["output_dir"], datetime.now().strftime("%Y-%m-%d_%I%p"))

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        bf16=cfg["bf16"],
        max_grad_norm=cfg.get("max_grad_norm", 0),
        seed=cfg["seed"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        report_to="none",
        max_steps=cfg.get("max_steps", -1),
        save_total_limit=cfg.get("save_total_limit", 3),
        # forward() takes **kwargs, so HF can't infer columns to keep -- keep them
        # all and let the collator consume query/pos/neg.
        remove_unused_columns=False,
        # the loss assumes a uniform chunk = bs / nb_gpus, so drop any final
        # partial batch that wouldn't split evenly across GPUs.
        dataloader_drop_last=True,
    )

    trainer = CSpRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    _neg_usage = cfg.get("loss_neg_usage")
    trainer.loss_fn = MatryoshkaContrastiveLoss(
        field_names=tuple(cfg["loss_field_names"]),
        temperature=cfg.get("temperature", 1.0),
        learnable_temperature=cfg.get("learnable_temperature", False),
        neg_usage=tuple(_neg_usage) if _neg_usage is not None else None,
    )

    model.loss_fn = trainer.loss_fn
    trainer.lambda_q = cfg["lambda_q"]
    trainer.lambda_d = cfg["lambda_d"]
    trainer.flops_warmup_steps = cfg.get("flops_warmup_steps", 0)


    os.makedirs(output_dir, exist_ok=True)

    trainer.train()

    hf_model = model.sparse_model.transformer.model
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # save learned combiner lambdas (combined-score training only)
    if hasattr(model, "current_lambdas"):
        with open(os.path.join(output_dir, "combiner_lambdas.json"), "w") as f:
            json.dump(model.current_lambdas(), f, indent=2)

    # save vocab partition
    with open(os.path.join(output_dir, "vocab_partitions.json"), "w") as f:
        json.dump(vocab_partitions, f, indent=2)
    # save a copy of the resolved config for reproducibility
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    
    print(f"Saved trained CSpR model to {output_dir}")


if __name__ == "__main__":
    main()
