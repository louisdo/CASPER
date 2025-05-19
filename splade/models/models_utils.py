import json
from omegaconf import DictConfig

from ..models.transformer_rep import Splade, SpladeDoc, SpladeTopK, SpladeLexical, SpladeMaxSim, PhraseSplade, PhraseSpladev2, PhraseSpladev3


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
        "phrase_splade_v3": PhraseSpladev3
    }
    try:
        model_class = model_map[config["matching_type"]]
    except KeyError:
        raise NotImplementedError("provide valid matching type ({})".format(config["matching_type"]))
    
    if model_class in ["phrase_splade"] and config["non_phrase_indices_path"] is not None:
        with open(config["non_phrase_indices_path"]) as f:
            non_phrase_indices = json.load(f)

        init_dict["non_phrase_indices"] = non_phrase_indices
        
    return model_class(**init_dict)
