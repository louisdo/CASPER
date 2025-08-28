from splade.models.transformer_rep import SpladeMaxSim, Splade, PhraseSpladev2, PhraseSpladev3, PhraseSpladev4, PhraseSpladev5

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
    
    "phrase_splade_58": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_58/debug/checkpoint/model",
    "phrase_splade_60": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_60/debug/checkpoint/model",
    "phrase_splade_61": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_61/debug/checkpoint/model",
    "phrase_splade_62": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_62/debug/checkpoint/model",
    "phrase_splade_63": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_63/debug/checkpoint/model",
    "phrase_splade_64": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_64/debug/checkpoint/model",
    "phrase_splade_65": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_65/debug/checkpoint/model",
    "phrase_splade_66": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_66/debug/checkpoint/model",
    "phrase_splade_67": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_67/debug/checkpoint/model",
    "phrase_splade_68": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_68/debug/checkpoint/model",
    "phrase_splade_69": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_69/debug/checkpoint/model",

    "phrase_splade_71": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_71/debug/checkpoint/model",
    "phrase_splade_72": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_72/debug/checkpoint/model",
    "phrase_splade_73": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_73/debug/checkpoint/model",
    "phrase_splade_74": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_74/debug/checkpoint/model",
    "phrase_splade_75": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_75/debug/checkpoint/model",
    "phrase_splade_76": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_76/debug/checkpoint/model",
    "phrase_splade_77": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_77/debug/checkpoint/model",
    "phrase_splade_78": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_78/debug/checkpoint/model",
    "phrase_splade_79": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_79/debug/checkpoint/model",
    "phrase_splade_80": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_80/debug/checkpoint/model",
    "phrase_splade_81": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_81/debug/checkpoint/model",

    "phrase_splade_83": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_83/debug/checkpoint/model",
    "phrase_splade_84": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_84/debug/checkpoint/model",
    "phrase_splade_85": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_85/debug/checkpoint/model",
    "phrase_splade_86": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_86/debug/checkpoint/model",

    "phrase_splade_87": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_87/debug/checkpoint/model",
    "phrase_splade_88": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_88/debug/checkpoint/model",
    "phrase_splade_89": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_89/debug/checkpoint/model",
    "phrase_splade_90": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_90/debug/checkpoint/model",
    "phrase_splade_91": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_91/debug/checkpoint/model",
    "phrase_splade_92": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_92/debug/checkpoint/model",
    "phrase_splade_93": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_93/debug/checkpoint/model",


    "phrase_splade_71_cfscube_taxoindex": "/scratch/lamdo/phrase_splade_checkpoints/phrase_splade_71_cfscube_taxoindex/debug/checkpoint/model",

    "splade_addedword_1": "/scratch/lamdo/phrase_splade_checkpoints/phrase_addedword_1/debug/checkpoint/model",
    "splade_addedword_2": "/scratch/lamdo/phrase_splade_checkpoints/phrase_addedword_2/debug/checkpoint/model",
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

    "phrase_splade_58": PhraseSpladev3,
    "phrase_splade_60": PhraseSpladev3,
    "phrase_splade_61": PhraseSpladev4,
    "phrase_splade_62": PhraseSpladev5,
    "phrase_splade_63": PhraseSpladev3,
    "phrase_splade_64": PhraseSpladev3,
    "phrase_splade_65": PhraseSpladev3,
    "phrase_splade_66": PhraseSpladev3,
    "phrase_splade_67": PhraseSpladev3,
    "phrase_splade_68": PhraseSpladev3,
    "phrase_splade_69": PhraseSpladev3,

    "phrase_splade_71": PhraseSpladev3,
    "phrase_splade_72": PhraseSpladev5,
    "phrase_splade_73": PhraseSpladev3,
    "phrase_splade_74": PhraseSpladev3,
    "phrase_splade_75": PhraseSpladev3,
    "phrase_splade_76": PhraseSpladev3,
    "phrase_splade_77": PhraseSpladev3,
    "phrase_splade_78": PhraseSpladev3,
    "phrase_splade_79": PhraseSpladev3,
    "phrase_splade_80": PhraseSpladev3,
    "phrase_splade_81": PhraseSpladev3,

    "phrase_splade_83": PhraseSpladev2,
    "phrase_splade_84": PhraseSpladev2,
    "phrase_splade_85": PhraseSpladev2,
    "phrase_splade_86": PhraseSpladev2,

    "phrase_splade_71_cfscube_taxoindex": PhraseSpladev3,
    "phrase_splade_87": PhraseSpladev3,
    "phrase_splade_88": PhraseSpladev3,
    "phrase_splade_89": PhraseSpladev3,
    "phrase_splade_90": PhraseSpladev3,
    "phrase_splade_91": PhraseSpladev3,
    "phrase_splade_92": PhraseSpladev3,
    "phrase_splade_93": PhraseSpladev3,

    "splade_addedword_1": Splade,
    "splade_addedword_2": PhraseSpladev3,
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

    "phrase_splade_58": False,
    "phrase_splade_60": False,
    "phrase_splade_61": False,
    "phrase_splade_62": False,
    "phrase_splade_63": False,
    "phrase_splade_64": False,
    "phrase_splade_65": False,
    "phrase_splade_66": False,
    "phrase_splade_67": False,
    "phrase_splade_68": False,
    "phrase_splade_69": False,

    "phrase_splade_71": False,
    "phrase_splade_72": False,
    "phrase_splade_73": False,
    "phrase_splade_74": False,
    "phrase_splade_75": False,
    "phrase_splade_76": False,
    "phrase_splade_77": False,
    "phrase_splade_78": False,
    "phrase_splade_79": False,
    "phrase_splade_80": False,
    "phrase_splade_81": False,

    "phrase_splade_83": False,
    "phrase_splade_84": False,
    "phrase_splade_85": False,
    "phrase_splade_86": False,


    "phrase_splade_87": False,
    "phrase_splade_88": False,
    "phrase_splade_89": False,
    "phrase_splade_90": False,
    "phrase_splade_91": False,
    "phrase_splade_92": False,
    "phrase_splade_93": False,


    "phrase_splade_71_cfscube_taxoindex": False,

    "splade_addedword_1": False,
    "splade_addedword_2": False,
    "eru_kg": False,
    "normal_splade_pretrains2orc": False,
    "splade_max_1": False
}


# if not specified, will default to 30522
model_name_2_original_bert_vocab_size = {
    "phrase_splade_83": 31090,
    "phrase_splade_86": 31090
}