import warnings
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from train.cspr.pmb.configuration_pmb import PartitionedModernBertConfig


class PartitionedModernBertForMaskedLM(ModernBertForMaskedLM):
    config_class = PartitionedModernBertConfig

    def __init__(self, config: PartitionedModernBertConfig):
        super().__init__(config)
        if config.vocab_partitions is not None:
            self.partition_names = list(config.vocab_partitions.keys())
            for name, indices in config.vocab_partitions.items():
                self.register_buffer(f"partition_{name}", torch.tensor(indices, dtype=torch.long))

        self.emb_projection = torch.nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size,
            bias=False,
        )
        self._init_weights(self.emb_projection)

        self.model.embeddings.tok_embeddings.weight.requires_grad_(False)

        if self._check_vocab_size() is False:
            warnings.warn(
                f"Embedding vocab size ({self.model.embeddings.tok_embeddings.num_embeddings}) "
                f"does not match total vocab_partitions size "
                f"({sum(len(p) for p in self.config.vocab_partitions.values())}). "
                "Did you forget to call add_embeddings?",
                UserWarning,
                stacklevel=2,
            )

    def _init_weights(self, module):
        if module is getattr(self, "emb_projection", None):
            torch.nn.init.eye_(module.weight)
        else:
            super()._init_weights(module)

    def _check_vocab_size(self) -> bool:
        total_vocab = sum(len(p) for p in self.config.vocab_partitions.values())
        return self.model.embeddings.tok_embeddings.num_embeddings == total_vocab
    

    @torch.no_grad()
    def add_embeddings(self, partition_embeddings: Dict[str, torch.Tensor]) -> None:
        if "token" in partition_embeddings:
            raise ValueError(
                "Cannot overwrite the 'token' partition: original token embeddings are frozen."
            )

        # Grow the embedding matrix to fit all partitions if it hasn't been resized yet.
        if not self._check_vocab_size():
            total_vocab = sum(len(p) for p in self.config.vocab_partitions.values())
            self.resize_token_embeddings(total_vocab)
            self.model.embeddings.tok_embeddings.weight.requires_grad_(False)

        tok_embeddings = self.model.embeddings.tok_embeddings
        hidden_size = tok_embeddings.embedding_dim

        for name, embeddings in partition_embeddings.items():
            if name not in self.partition_names:
                raise ValueError(
                    f"Unknown partition '{name}'. Expected one of {self.partition_names}."
                )

            indices = getattr(self, f"partition_{name}")  # [partition_size], long
            expected_shape = (indices.numel(), hidden_size)
            if tuple(embeddings.shape) != expected_shape:
                raise ValueError(
                    f"Embedding tensor for partition '{name}' has shape {tuple(embeddings.shape)}, "
                    f"expected {expected_shape}."
                )

            tok_embeddings.weight[indices] = embeddings.to(
                dtype=tok_embeddings.weight.dtype,
                device=tok_embeddings.weight.device,
            )
        


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        partitioned: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor] | MaskedLMOutput:
        raw_embeds = self.model.embeddings(input_ids = input_ids) # [B, S, d]

        projected = self.emb_projection(raw_embeds) # [B, S, d]
        # projected = raw_embeds

        outputs = self.model(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # [B, S, d]

        head_output = self.head(hidden_states)  # [B, S, d]

        head_proj = head_output @ self.emb_projection.weight  # [B, S, d]
        # head_proj = head_output
        vocab_weight = self.model.embeddings.tok_embeddings.weight  # [V, d], frozen
        logits = F.linear(head_proj, vocab_weight, bias = self.decoder.bias)  # [B, S, V]

        if partitioned:
            return {name: logits[:, :, getattr(self, f"partition_{name}")] \
                    for name in self.partition_names}

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
