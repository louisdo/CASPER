import argparse
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Trainer


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples_path", type=str)
    parser.add_argument("--queries_path", type=str)
    parser.add_argument("--collection_path", type=str)
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--checkpoint", type = str, default='distilbert-base-uncased')
    parser.add_argument("--experiment_folder", type = str, required = True)
    parser.add_argument("--experiment_name", type = str, default = "colbert")

    args = parser.parse_args()

    triples = args.triples_path
    queries = args.queries_path
    collection = args.collection_path
    nranks = args.nranks

    batch_size = args.batch_size

    checkpoint = args.checkpoint
    experiment_folder = args.experiment_folder
    experiment_name = args.experiment_name

    # use 4 gpus (e.g. four A100s, but you can use fewer by changing nway,accumsteps,bsize).
    with Run().context(RunConfig(nranks=nranks, experiment = experiment_name, root = experiment_folder)):
        config = ColBERTConfig(
            bsize=batch_size,
            query_maxlen=64,
            doc_maxlen=256,
            use_ib_negatives = True
        )
        trainer = Trainer(triples=triples, queries=queries, collection=collection, config=config)

        checkpoint_path = trainer.train(checkpoint)  # or start from scratch, like `bert-base-uncased`
        print(f"Saved checkpoint to {checkpoint_path}...")



if __name__ == "__main__":
    train()