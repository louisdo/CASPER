import torch
from argparse import ArgumentParser
from data_collator import FocusedMaskDataCollator
from scirepeval_dataset import SciRepEvalDataset
from s2orc_dataset import S2ORCDataset
from s2orc_cs_dataset import S2ORCCSDataset
from erukg_dataset import ERUKGDataset
from msmarco_dataset import MSMARCODataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments
)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = ArgumentParser()
    parser.add_argument("--collection_path", type = str, required = True)
    parser.add_argument("--input_model_name", type = str, required = True)
    parser.add_argument("--special_token_id_start", type = int, default = 30522)
    parser.add_argument("--mlm_probability", type = float, default = 0.15)
    parser.add_argument("--max_collection", type = int, default = 1000000000)
    parser.add_argument("--batch_size", type = int, default= 16)
    parser.add_argument("--output_dir", type = str, required=True)
    parser.add_argument("--num_train_epochs", type = int, default = 2)
    parser.add_argument("--max_steps", type = int, default = 70000)
    parser.add_argument("--save_steps", type = int, default = 10000)
    parser.add_argument("--save_total_limit", type = int, default = 2)
    parser.add_argument("--learning_rate", type = float, default = 5e-5)
    parser.add_argument("--weight_decay", type = float, default = 0.01)
    parser.add_argument("--special_token_mask_probability", type = float, default = 0.9)
    parser.add_argument("--dataset_name", type = str, default = "scirepeval")
    parser.add_argument("--fp16", type = str2bool, default = False)

    args = parser.parse_args()


    collection_path = args.collection_path
    input_model_name = args.input_model_name
    special_token_id_start = args.special_token_id_start
    mlm_probability = args.mlm_probability
    max_collection = args.max_collection
    batch_size = args.batch_size
    output_dir = args.output_dir
    num_train_epochs = args.num_train_epochs
    max_steps = args.max_steps
    save_steps = args.save_steps
    save_total_limit = args.save_total_limit
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    special_token_mask_probability = args.special_token_mask_probability
    dataset_name = args.dataset_name
    fp16 = args.fp16

    dataset_name_2_dataset_class = {
        "scirepeval": SciRepEvalDataset,
        "s2orc": S2ORCDataset, # s2orc open-domain corpus
        "s2orc_cs": S2ORCCSDataset, # s2orc computer science corpus
        "erukgds": ERUKGDataset,
        "msmarco": MSMARCODataset
    }


    model_name = input_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    data_collator = FocusedMaskDataCollator(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        special_token_id_start = special_token_id_start,
        special_token_mask_probability=special_token_mask_probability
    )

    custom_dataset = dataset_name_2_dataset_class[dataset_name](
        path = collection_path, 
        tokenizer=tokenizer, 
        max_collections=max_collection
    )



    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        max_steps = max_steps,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=True,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir="./logs",
        logging_steps=200,
        report_to="none",
        adam_beta1=0.9,  
        adam_beta2=0.999,  
        lr_scheduler_type="linear", 
        warmup_steps=2500 if dataset_name == "erukgds" else 10000,
        fp16 = fp16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=custom_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()