import os, sys
sys.path.append("../..")
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from model_name_2_model_info import model_name_2_model_dir
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--experiment_path", type = str, default = "./experiments")
    parser.add_argument("--eval_data_folder", type = str, default="../../data")

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    experiment_path = args.experiment_path
    eval_data_folder = args.eval_data_folder

    model_checkpoint = model_name_2_model_dir[model_name]

    collection_path = os.path.join(f"{eval_data_folder}/{dataset_name}/collection.jsonl")


    with Run().context(RunConfig(nranks=1, root=experiment_path, experiment=dataset_name)):

        config = ColBERTConfig(
            nbits=2,
            query_maxlen=256,
            doc_maxlen=256,
            mask_punctuation=True
        )
        indexer = Indexer(checkpoint=model_checkpoint, config=config, verbose=True)
        indexer.index(name=f"{dataset_name}.nbits=2", collection=collection_path, overwrite=True)

if __name__=='__main__':

    main()