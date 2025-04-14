from splade.models.transformer_rep import SpladeMaxSim, Splade

model_name_2_path = {
    "splade_normal": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal/debug/checkpoint/model",
    "splade_maxsim": "/scratch/lamdo/splade_maxsim_ckpts/testing/debug/checkpoint/model",
    "original_spladev2": "naver/splade_v2_distil",
    "original_spladev2_max": "naver/splade_v2_max",
    "splade_cocondenser_ensembledistil": "naver/splade-cocondenser-ensembledistil",
    "splade_normal_100k": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal_100k/debug/checkpoint/model",
    "splade_maxsim_100k": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_100k/debug/checkpoint/model",
    "splade_normal_100k_lowreg": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal_100k_lowreg/debug/checkpoint/model",
    "splade_maxsim_100k_lowreg": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_100k_lowreg/debug/checkpoint/model",
    "splade_maxsim_100k_lowregv2": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_100k_lowregv2/debug/checkpoint/model",
    "splade_normal_150k_lowreg": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal_150k_lowreg/debug/checkpoint/model",
    "splade_maxsim_150k_lowregv2": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_150k_lowregv2_/debug/checkpoint/model",
    "splade_maxsim_150k_lowregv3": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_150k_lowregv3/debug/checkpoint/model",
    "splade_maxsim_150k_lowregv4": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_150k_lowregv4/debug/checkpoint/model",
    "splade_normal_150k_lowreg_ensembledistil": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal_150k_lowreg_ensembledistil/debug/checkpoint/model",
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_10/debug/checkpoint/model", # first version, 4.4k phrases added
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_11/debug/checkpoint/model", # second version, 14k phrases added 
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_12/debug/checkpoint/model", # third version, 16k added from s2orc, pretraining on s2orc
    "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_13/debug/checkpoint/model", # fourth version, 16k added from s2orc, pretraining on s2orc, lower regularization than third version
    "eru_kg": "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8-1/debug/checkpoint/model",
    "normal_splade_pretrains2orc": "/scratch/lamdo/phrase_splade_checkpoints/normal_splade_pretrains2orc/debug/checkpoint/model"
}

model_name_2_model_class = {
    "splade_normal": Splade,
    "splade_maxsim": SpladeMaxSim,
    "original_spladev2": Splade,
    "original_spladev2_max": Splade,
    "splade_cocondenser_ensembledistil": Splade,
    "splade_normal_100k": Splade,
    "splade_maxsim_100k": SpladeMaxSim,
    "splade_normal_100k_lowreg": Splade,
    "splade_maxsim_100k_lowreg": SpladeMaxSim,
    "splade_maxsim_100k_lowregv2": SpladeMaxSim,
    "splade_normal_150k_lowreg": Splade,
    "splade_maxsim_150k_lowregv2": SpladeMaxSim,
    "splade_maxsim_150k_lowregv3": SpladeMaxSim,
    "splade_maxsim_150k_lowregv4": SpladeMaxSim,
    "splade_normal_150k_lowreg_ensembledistil": Splade,
    "phrase_splade": Splade,
    "eru_kg": Splade,
    "normal_splade_pretrains2orc": Splade
}

model_name_2_is_maxsim = {
    "splade_normal": False,
    "splade_maxsim": True,
    "original_spladev2": False,
    "original_spladev2_max": False,
    "splade_cocondenser_ensembledistil": False,
    "splade_normal_100k": False,
    "splade_maxsim_100k": True,
    "splade_normal_100k_lowreg": False,
    "splade_maxsim_100k_lowreg": True,
    "splade_maxsim_100k_lowregv2": True,
    "splade_normal_150k_lowreg": False,
    "splade_maxsim_150k_lowregv2": True,
    "splade_maxsim_150k_lowregv3": True,
    "splade_maxsim_150k_lowregv4": True,
    "splade_normal_150k_lowreg_ensembledistil": False,
    "phrase_splade": False,
    "eru_kg": False,
    "normal_splade_pretrains2orc": False
}