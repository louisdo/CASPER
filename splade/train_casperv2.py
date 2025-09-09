import os

import hydra
import torch
from omegaconf import DictConfig, open_dict
from torch.utils import data

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader, CASPERv2SiamesePairsDataLoader
from .datasets.datasets import CASPERv2PairsDatasetPreload, \
    CollectionDatasetPreLoad
from .losses.regularization import init_regularizer, RegWeightScheduler
from .models.models_utils import get_model
from .optim.bert_optim import init_simple_bert_optim
from .tasks.transformer_evaluator import SparseApproxEvalWrapper
from .tasks.transformer_trainer_casper import SiameseCASPERv2Trainer
from .utils.utils import set_seed, restore_model, get_initialize_config, get_loss, set_seed_from_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def train(exp_dict: DictConfig):
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    model = get_model(config, init_dict)
    random_seed = set_seed_from_config(config)

    optimizer, scheduler = init_simple_bert_optim(model, lr=config["lr"], warmup_steps=config["warmup_steps"],
                                                  weight_decay=config["weight_decay"],
                                                  num_training_steps=config["nb_iterations"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ################################################################
    # CHECK IF RESUME TRAINING
    ################################################################
    iterations = (1, config["nb_iterations"] + 1)  # tuple with START and END
    regularizer = None
    if os.path.exists(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar")):
        print("@@@@ RESUMING TRAINING @@@")
        print("WARNING: change seed to change data order when restoring !")
        set_seed(random_seed + 666)
        if device == torch.device("cuda"):
            ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"))
        else:
            ckpt = torch.load(os.path.join(config["checkpoint_dir"], "model_ckpt/model_last.tar"), map_location=device)
        print("starting from step", ckpt["step"])
        print("{} remaining iterations".format(iterations[1] - ckpt["step"]))
        iterations = (ckpt["step"] + 1, config["nb_iterations"])
        restore_model(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if device == torch.device("cuda"):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "regularizer" in ckpt:
            print("loading regularizer")
            regularizer = ckpt.get("regularizer", None)

    if torch.cuda.device_count() > 1:
        print(" --- use {} GPUs --- ".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    loss = get_loss(config)

    # initialize regularizer dict
    if "regularizer" in config and regularizer is None:  # else regularizer is loaded
        output_dim = model.module.output_dim if hasattr(model, "module") else model.output_dim
        regularizer = {"eval": {"L0": {"loss": init_regularizer("L0")},
                                "sparsity_ratio": {"loss": init_regularizer("sparsity_ratio",
                                                                            output_dim=output_dim)}},
                       "train": {}}
        if config["regularizer"] == "eval_only":
            # just in the case we train a model without reg but still want the eval metrics like L0
            pass
        else:
            for reg in config["regularizer"]:
                temp = {"loss": init_regularizer(config["regularizer"][reg]["reg"]),
                        "targeted_rep": config["regularizer"][reg]["targeted_rep"]}
                d_ = {}
                if "lambda_q" in config["regularizer"][reg]:
                    d_["lambda_q"] = RegWeightScheduler(config["regularizer"][reg]["lambda_q"],
                                                        config["regularizer"][reg]["T"])
                if "lambda_d" in config["regularizer"][reg]:
                    d_["lambda_d"] = RegWeightScheduler(config["regularizer"][reg]["lambda_d"],
                                                        config["regularizer"][reg]["T"])
                temp["lambdas"] = d_  # it is possible to have reg only on q or d if e.g. you only specify lambda_q
                # in the reg config
                # targeted_rep is just used to indicate which rep to constrain (if e.g. the model outputs several
                # representations)
                # the common case: model outputs "rep" (in forward) and this should be the value for this targeted_rep
                regularizer["train"][reg] = temp

    # fix for current in batch neg losses that break on last batch
    if config["loss"] in ("InBatchNegHingeLoss", "InBatchPairwiseNLL"):
        drop_last = True
    else:
        drop_last = False

    if exp_dict["data"].get("type", "") == "triplets":
        data_train = CASPERv2PairsDatasetPreload(data_dir=exp_dict["data"]["TRAIN_DATA_DIR"])
        train_mode = "triplets"
    elif exp_dict["data"].get("type", "") == "triplets_with_distil":
        raise NotImplementedError
    elif exp_dict["data"].get("type", "") == "hard_negatives":
        raise NotImplementedError
    else:
        raise ValueError("provide valid data type for training")

    val_loss_loader = None  # default
    if "VALIDATION_SIZE_FOR_LOSS" in exp_dict["data"]:
        print("initialize loader for validation loss")
        print("split train, originally {} pairs".format(len(data_train)))
        data_train, data_val = torch.utils.data.random_split(data_train, lengths=[
            len(data_train) - exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"],
            exp_dict["data"]["VALIDATION_SIZE_FOR_LOSS"]])
        print("train: {} pairs ~~ val: {} pairs".format(len(data_train), len(data_val)))
        if train_mode == "triplets":
            val_loss_loader = CASPERv2SiamesePairsDataLoader(dataset=data_val, batch_size=config["eval_batch_size"],
                                                     shuffle=False,
                                                     num_workers=4,
                                                     tokenizer_type=config["tokenizer_type"],
                                                     max_length=config["max_length"], drop_last=drop_last)
        else:
            raise NotImplementedError("This option is currently not supported for CASPERv2")

    if train_mode == "triplets":
        train_loader = CASPERv2SiamesePairsDataLoader(dataset=data_train, batch_size=config["train_batch_size"], shuffle=True,
                                              num_workers=4,
                                              tokenizer_type=config["tokenizer_type"],
                                              max_length=config["max_length"], drop_last=drop_last)
    else:
        raise NotImplementedError("This option is currently not supported for CASPERv2")

    val_evaluator = None
    if "VALIDATION_FULL_RANKING" in exp_dict["data"]:
        raise NotImplementedError

    # #################################################################
    # # TRAIN
    # #################################################################
    print("+++++ BEGIN TRAINING +++++")
    trainer = SiameseCASPERv2Trainer(model=model, iterations=iterations, loss=loss, optimizer=optimizer,
                                    config=config, scheduler=scheduler,
                                    train_loader=train_loader, validation_loss_loader=val_loss_loader,
                                    validation_evaluator=val_evaluator,
                                    regularizer=regularizer)
    trainer.train()


if __name__ == "__main__":
    train()
