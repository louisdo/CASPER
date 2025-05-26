from splade.models.transformer_rep import SpladeMaxSim, Splade, PhraseSpladev3

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
    "splade_maxsim_150k_lowregv5": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_150k_lowregv5/debug/checkpoint/model", # similar to v4, but lower regularization
    "splade_maxsim_100k_lowregv6": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_100k_lowregv6/debug/checkpoint/model",
    "splade_maxsim_150k_lowregv6": "/scratch/lamdo/splade_maxsim_ckpts/splade_maxsim_150k_lowregv6/debug/checkpoint/model",
    "splade_normal_150k_lowreg_ensembledistil": "/scratch/lamdo/splade_maxsim_ckpts/splade_normal_150k_lowreg_ensembledistil/debug/checkpoint/model",
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_10/debug/checkpoint/model", # first version, 4.4k phrases added
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_11/debug/checkpoint/model", # second version, 14k phrases added 
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_12/debug/checkpoint/model", # third version, 16k added from s2orc, pretraining on s2orc
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_13/debug/checkpoint/model", # fourth version, 16k added from s2orc, pretraining on s2orc, lower regularization than third version
    # "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_24/debug/checkpoint/model", # fifth version, 16k added from s2orc, pretraining on s2orc, trained on kp20k, lower regularization
    "phrase_splade": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_25/debug/checkpoint/model", # sixth version, similar to fifth, but switching back and fourth between tokenization using tokens and phrases during training
    "phrase_splade_27": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_27/debug/checkpoint/model", # seventh version, similar to sixth, training with ERU-KG dataset, high regularization
    "phrase_splade_26": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_26/debug/checkpoint/model", # eighth version, similar to seventh, training with ERU-KG dataset, low regularization
    "phrase_splade_24": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_24/debug/checkpoint/model", # same as phrase_splade_25, but no switching back and fourth
    "phrase_splade_31": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_31/debug/checkpoint/model",
    "phrase_splade_33": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_33/debug/checkpoint/model",
    "phrase_splade_34": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_34/debug/checkpoint/model",
    "phrase_splade_35": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_35/debug/checkpoint/model",
    "phrase_splade_36": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_36/debug/checkpoint/model",
    "phrase_splade_37": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_37/debug/checkpoint/model",
    "phrase_splade_38": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_38/debug/checkpoint/model",
    "phrase_splade_39": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_39/debug/checkpoint/model", # pretrain from distilbert (prioritize phrase), new dataset from s2orc (custom loss)
    "phrase_splade_40": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_40/debug/checkpoint/model", # same with 39, but with bert
    "phrase_splade_41": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_41/debug/checkpoint/model", # pretrain from distilbert (phrase and tokens are pretrained equally), new dataset from s2orc (normal splade loss)
    "phrase_splade_42": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_42/debug/checkpoint/model", # pretrain from distilbert (phrase and tokens are pretrained equally), new dataset from s2orc (custom loss)
    "phrase_splade_43": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_43/debug/checkpoint/model",
    "phrase_splade_44": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_44/debug/checkpoint/model",
    "phrase_splade_45": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_45/debug/checkpoint/model", # # pretrain from distilbert (prioritize phrase) with s2orc, 30k phrases chosen from s2orc (phrase vocab bulding algorithm -- maximum coverage), SPLADE training with our new dataset
    "phrase_splade_46": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_46/debug/checkpoint/model", # # pretrain from distilbert (prioritize phrase) with msmarco, 30k phrases chosen from msmarco (phrase vocab bulding algorithm -- maximum coverage), SPLADE training with msmarco
    "phrase_splade_47": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_47/debug/checkpoint/model", # pretty much the same as 45, with adjusted loss function and regularization
    "phrase_splade_48": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_48/debug/checkpoint/model", # similar to 46 but with regular SPLADE loss and not our custom loss
    "phrase_splade_49": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_49/debug/checkpoint/model",
    "phrase_splade_50": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_50/debug/checkpoint/model",
    "phrase_splade_51": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_51/debug/checkpoint/model", # similar to 45, but without weight in loss (cl_tokens + cl_phrases + reg_total)
    "phrase_splade_52": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_52/debug/checkpoint/model", # new architecture, max pooling for token and mean pooling for concepts
    "phrase_splade_53": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_53/debug/checkpoint/model",
    "phrase_splade_54": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_54/debug/checkpoint/model",
    "phrase_splade_55": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_55/debug/checkpoint/model", # same as model 52, but using a different pretrain checkpoint, with phrase vocabulary is chosen based on frequency instead of the vocab building algorithm
    "phrase_splade_56": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_56/debug/checkpoint/model", # same as model 52, but using a different pretrain checkpoint (60k phrases)
    "splade_addedword_1": "/scratch/lamdo/phrase_splade_checkpoints/phrase_addedword_1/debug/checkpoint/model",
    "eru_kg": "/scratch/lamdo/splade_checkpoints/experiments_combined_references_v8-1/debug/checkpoint/model",
    "normal_splade_pretrains2orc": "/scratch/lamdo/phrase_splade_checkpoints/normal_splade_pretrains2orc/debug/checkpoint/model",
    "splade_max_1": "/scratch/lamdo/phrase_splade_checkpoints/splade_max_1/debug/checkpoint/model"
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
    "splade_maxsim_150k_lowregv5": SpladeMaxSim,
    "splade_maxsim_100k_lowregv6": SpladeMaxSim,
    "splade_maxsim_150k_lowregv6": Splade,
    "splade_normal_150k_lowreg_ensembledistil": Splade,
    "phrase_splade": Splade,
    "phrase_splade_26": Splade,
    "phrase_splade_27": Splade,
    "phrase_splade_24": Splade,
    "phrase_splade_31": Splade,
    "phrase_splade_33": Splade,
    "phrase_splade_34": Splade,
    "phrase_splade_35": Splade,
    "phrase_splade_36": Splade,
    "phrase_splade_37": Splade,
    "phrase_splade_38": Splade,
    "phrase_splade_39": Splade,
    "phrase_splade_40": Splade,
    "phrase_splade_41": Splade,
    "phrase_splade_42": Splade,
    "phrase_splade_43": Splade,
    "phrase_splade_44": Splade,
    "phrase_splade_45": Splade,
    "phrase_splade_46": Splade,
    "phrase_splade_47": Splade,
    "phrase_splade_48": Splade,
    "phrase_splade_49": Splade,
    "phrase_splade_50": Splade,
    "phrase_splade_51": Splade,
    "phrase_splade_52": PhraseSpladev3,
    "phrase_splade_53": PhraseSpladev3,
    "phrase_splade_54": PhraseSpladev3,
    "phrase_splade_55": PhraseSpladev3,
    "phrase_splade_56": PhraseSpladev3,
    "splade_addedword_1": Splade,
    "eru_kg": Splade,
    "normal_splade_pretrains2orc": Splade,
    "splade_max_1": Splade
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
    "splade_maxsim_150k_lowregv5": True,
    "splade_maxsim_100k_lowregv6": True,
    "splade_maxsim_150k_lowregv6": False,
    "splade_normal_150k_lowreg_ensembledistil": False,
    "phrase_splade": False,
    "phrase_splade_26": False,
    "phrase_splade_27": False,
    "phrase_splade_24": False,
    "phrase_splade_31": False,
    "phrase_splade_33": False,
    "phrase_splade_34": False,
    "phrase_splade_35": False,
    "phrase_splade_36": False,
    "phrase_splade_37": False,
    "phrase_splade_38": False,
    "phrase_splade_39": False,
    "phrase_splade_40": False,
    "phrase_splade_41": False,
    "phrase_splade_42": False,
    "phrase_splade_43": False,
    "phrase_splade_44": False,
    "phrase_splade_45": False,
    "phrase_splade_46": False,
    "phrase_splade_47": False,
    "phrase_splade_48": False,
    "phrase_splade_49": False,
    "phrase_splade_50": False,
    "phrase_splade_51": False,
    "phrase_splade_52": False,
    "phrase_splade_53": False,
    "phrase_splade_54": False,
    "phrase_splade_55": False,
    "phrase_splade_56": False,
    "splade_addedword_1": False,
    "eru_kg": False,
    "normal_splade_pretrains2orc": False,
    "splade_max_1": False
}