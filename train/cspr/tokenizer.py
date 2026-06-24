import json
import os
from typing import Optional, Dict, List
from collections import Counter
from flashtext import KeywordProcessor
from transformers import AutoTokenizer, PreTrainedTokenizer


def format_keyphrase(keyphrase: str):
    words = keyphrase.split()
    return f"""<<{"_".join(words)}>>"""


class TextProcessor:
    def __init__(self, keyphrase_vocab_path: str, sep_token: str = "[SEP]", max_keyphrase: int = 15000):
        with open(keyphrase_vocab_path) as f:
            self.keyphrase_vocab = json.load(f)

        self.sep_token = sep_token
        self.keyphrase_vocab = self.keyphrase_vocab[:max_keyphrase]

        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        for canonical, variations in self.keyphrase_vocab:
            token = format_keyphrase(canonical)
            for surface in [canonical, *variations]:
                self.keyword_processor.add_keyword(surface, token)

    def process_each_text(self, text: str, add_negatives: bool = True) -> str | None:
        concept_counter = Counter(self.keyword_processor.extract_keywords(text))
        pos = {k: v for k, v in concept_counter.items() if "_" in k or v > 1}
        pos_keyphrase_tokens = list(dict.fromkeys(pos))
        if not pos_keyphrase_tokens:
            return None

        ret = f"{text}{self.sep_token}{''.join(pos_keyphrase_tokens)}"

        if add_negatives:
            neg_keyphrase_tokens = [k for k in dict.fromkeys(concept_counter) if k not in pos]
            if neg_keyphrase_tokens:
                ret += f"||NEG||{''.join(neg_keyphrase_tokens)}"

        return ret
    
    def process_each_text_inplace(self, text: str, keep_surface: bool = False) -> str:
        matches = self.keyword_processor.extract_keywords(text, span_info=True)

        concept_counter = Counter(token for token, _, _ in matches)
        pos = {k for k, v in concept_counter.items() if "_" in k or v > 1}
        if not pos:
            return text

        parts, cursor = [], 0
        for token, start, end in sorted(matches, key=lambda m: m[1]):
            if token not in pos:
                continue
            parts.append(text[cursor:start])
            parts.append(f"{text[start:end]} {token}" if keep_surface else token)
            cursor = end
        parts.append(text[cursor:])

        return "".join(parts)


class MaskedTextProcessor(TextProcessor):
    """Variant of TextProcessor that can pad the keyphrase segment with [MASK]
    tokens (or emit only [MASK] tokens), and returns that segment on its own so
    the caller can keep it out of length truncation.

    TextProcessor itself is left untouched.
    """

    def __init__(self, keyphrase_vocab_path: str, sep_token: str = "[SEP]",
                 mask_token: str = "[MASK]", max_keyphrase: int = 15000):
        super().__init__(keyphrase_vocab_path, sep_token=sep_token, max_keyphrase=max_keyphrase)
        self.mask_token = mask_token

    def build_keyphrase_segment(self, text: str, num_slots: int | None = None,
                                mask_only: bool = False) -> str:
        """Return the segment to append after the text (no leading [SEP])."""
        # Only [MASK] tokens, skip keyphrase detection entirely.
        if mask_only:
            return self.mask_token * (num_slots or 0)

        concept_counter = Counter(self.keyword_processor.extract_keywords(text))
        pos = {k: v for k, v in concept_counter.items() if "_" in k or v > 1}
        keyphrase_tokens = list(dict.fromkeys(pos))

        # Pad with [MASK] so the segment always holds `num_slots` tokens.
        if num_slots and len(keyphrase_tokens) < num_slots:
            keyphrase_tokens += [self.mask_token] * (num_slots - len(keyphrase_tokens))

        return "".join(keyphrase_tokens)


class CSpRTokenizer:
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 keyphrase_vocab_path: str,
                 original_bert_vocab_size: int,
                 num_keyphrase_slots: int = 5,
                 mask_only: bool = False,
                 mode: str = "append",
                 keep_surface: bool = False,
                 use_default: bool = False):
        if mode not in ("append", "inplace"):
            raise ValueError(f"Unknown mode '{mode}'. Expected 'append' or 'inplace'.")
        self.tokenizer = tokenizer
        self.num_keyphrase_slots = num_keyphrase_slots
        self.mask_only = mask_only
        # When True, bypass all keyphrase processing and behave exactly like the
        # underlying AutoTokenizer (no inline rewriting, no appended segment).
        self.use_default = use_default
        # "append": keyphrases/[MASK] after [SEP] as a text_pair (segment kept out
        # of truncation). "inplace": keyphrase tokens rewritten where they occur in
        # the text, tokenized as a single sequence.
        self.mode = mode
        self.keep_surface = keep_surface

        # create text processor here
        self.text_processor = MaskedTextProcessor(
            keyphrase_vocab_path = keyphrase_vocab_path,
            sep_token = self.tokenizer.sep_token,
            mask_token = self.tokenizer.mask_token,
            max_keyphrase = len(self.tokenizer) - original_bert_vocab_size
        )

    def __getattr__(self, name):
        return getattr(self.__dict__["tokenizer"], name)

    def __len__(self):
        # __getattr__ is not consulted for dunders, so delegate explicitly.
        return len(self.tokenizer)

    def __call__(self, text=None, **kwargs):
        if text is None:
            text = kwargs.pop("text", None)
        if text is None:
            raise ValueError("`text` is required")

        if self.use_default:
            # Plain tokenization, no keyphrase rewriting or appended segment.
            kwargs.setdefault("truncation", True)
            kwargs.setdefault("max_length", self.tokenizer.model_max_length)
            return self.tokenizer(text, **kwargs)

        if self.mode == "inplace":
            # Rewrite keyphrases inline and tokenize as a single sequence; no
            # segment to protect, so standard truncation applies.
            def rewrite(t: str) -> str:
                return self.text_processor.process_each_text_inplace(t, keep_surface=self.keep_surface)

            processed = rewrite(text) if isinstance(text, str) else [rewrite(t) for t in text]
            kwargs.setdefault("truncation", True)
            kwargs.setdefault("max_length", self.tokenizer.model_max_length)
            return self.tokenizer(processed, **kwargs)

        # append mode
        def segment(t: str) -> str:
            return self.text_processor.build_keyphrase_segment(
                t,
                num_slots=self.num_keyphrase_slots,
                mask_only=self.mask_only,
            )

        if isinstance(text, str):
            text_pair = segment(text)
        else:
            text_pair = [segment(t) for t in text]


        # Force only_first so the keyphrase/[MASK] segment is never truncated,
        # even if the caller passes truncation=True (e.g. the data collator).
        kwargs["truncation"] = "only_first"
        kwargs.setdefault("max_length", self.tokenizer.model_max_length)
        kwargs.setdefault("return_token_type_ids", False)

        return self.tokenizer(text, text_pair=text_pair, **kwargs)
    
    