import json, os
from omegaconf import DictConfig

from peft import LoraConfig, get_peft_model, TaskType

from ..models.transformer_rep import Splade, SpladeDoc, SpladeTopK, SpladeLexical, SpladeMaxSim, PhraseSplade, PhraseSpladev2, PhraseSpladev3, \
    PhraseSpladev4, PhraseSpladev5, PhraseSpladev3_2, CASPERv2


def get_model(config: DictConfig, init_dict: DictConfig):
    # no need to reload model here, it will be done later
    # (either in train.py or in Evaluator.__init__()

    model_map = {
        "splade": Splade,
        "splade_doc": SpladeDoc,
        "splade_topk": SpladeTopK,
        "splade_lexical": SpladeLexical,
        "splade_maxsim": SpladeMaxSim,
        "phrase_splade": PhraseSplade,
        "phrase_splade_v2": PhraseSpladev2,
        "phrase_splade_v3": PhraseSpladev3,
        "phrase_splade_v3_2": PhraseSpladev3_2,
        "phrase_splade_v4": PhraseSpladev4,
        "phrase_splade_v5": PhraseSpladev5,
        "casperv2": CASPERv2
    }
    try:
        model_class = model_map[config["matching_type"]]
    except KeyError:
        raise NotImplementedError("provide valid matching type ({})".format(config["matching_type"]))
    
    if model_class in ["phrase_splade"] and config["non_phrase_indices_path"] is not None:
        with open(config["non_phrase_indices_path"]) as f:
            non_phrase_indices = json.load(f)

        init_dict["non_phrase_indices"] = non_phrase_indices

    model = model_class(**init_dict)
    # if config["lora"]["apply_lora"]:
    #     attach_lora_to_casper(model,
    #                           r = config["lora"]["r"],
    #                           alpha = config["lora"]["alpha"],
    #                           dropout=config["lora"]["dropout"],
    #                           target_modules=config["lora"]["target_modules"])
        

    return model
        



def attach_lora_to_casper(model, r=8, alpha=16, dropout=0.05,
                          target_modules=("query","key","value","dense"),
                          task_type="masked_lm"):
    """
    model: an instance of CASPERv2 (or any of your classes that wrap AutoModelForMaskedLM)
    Apply LoRA to the encoder(s). Call this BEFORE creating the optimizer.
    """

    ttype = getattr(TaskType, "MASKED_LM", TaskType.FEATURE_EXTRACTION)  # fallback if older peft
    if task_type == "feature_extraction":
        ttype = TaskType.FEATURE_EXTRACTION

    peft_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=ttype,
        target_modules=list(target_modules)
    )

    model.transformer_rep.transformer = get_peft_model(
        model.transformer_rep.transformer, peft_cfg
    )
    model.transformer_rep.transformer.print_trainable_parameters()

    if getattr(model, "transformer_rep_q", None) is not None:
        model.transformer_rep_q.transformer = get_peft_model(
            model.transformer_rep_q.transformer, peft_cfg
        )


    return model