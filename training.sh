export CUDA_VISIBLE_DEVICES="0"
python training.py \
    --triples_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/triples.train.jsonl \
    --queries_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/queries.train.tsv \
    --collection_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/corpus.train.tsv \
    --nranks 1 \
    --batch_size 24 \
    --experiment_folder /scratch/lamdo/colbert/experiments \
    --experiment_name combined_cc+cocit+kp1m+query+title 



export CUDA_VISIBLE_DEVICES=2
python training.py \
    --triples_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/triples.train.jsonl \
    --queries_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/queries.train.tsv \
    --collection_path /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title/colbert_training_format/corpus.train.tsv \
    --nranks 1 \
    --batch_size 20 \
    --experiment_folder /scratch/lamdo/colbert/experiments \
    --experiment_name combined_cc+cocit+kp1m+query+title_v2 \
    --checkpoint lamdo/distilbert-s2orc-mlm-80000steps

CUDA_VISIBLE_DEVICES=2 python training.py \
--triples_path /scratch/lamdo/phrase_splade_datasets/combined_cc_cs+cocit_cs+kp20k+query_cs+title_cs/colbert_training_format/triples.train.jsonl \
--queries_path /scratch/lamdo/phrase_splade_datasets/combined_cc_cs+cocit_cs+kp20k+query_cs+title_cs/colbert_training_format/queries.train.tsv \
--collection_path /scratch/lamdo/phrase_splade_datasets/combined_cc_cs+cocit_cs+kp20k+query_cs+title_cs/colbert_training_format/corpus.train.tsv \
--nranks 1 \
--batch_size 20 \
--experiment_folder /scratch/lamdo/colbert/experiments \
--experiment_name combined_cc_cs+cocit_cs+kp20k+query_cs+title_cs



(query_token_id: str = DefaultVal("[unused0]"), 
doc_token_id: str = DefaultVal("[unused1]"), 
query_token: str = DefaultVal("[Q]"), 
doc_token: str = DefaultVal("[D]"), 
ncells: int = DefaultVal(None), 
centroid_score_threshold: float = DefaultVal(None), 
ndocs: int = DefaultVal(None), 
load_index_with_mmap: bool = DefaultVal(False), 
index_path: str = DefaultVal(None), 
index_bsize: int = DefaultVal(64), 
nbits: int = DefaultVal(1), 
kmeans_niters: int = DefaultVal(4), 
resume: bool = DefaultVal(False), 
pool_factor: int = DefaultVal(1), 
clustering_mode: str = DefaultVal("hierarchical"), 
protected_tokens: int = DefaultVal(0), 
similarity: str = DefaultVal("cosine"), 
bsize: int = DefaultVal(32), 
accumsteps: int = DefaultVal(1), 
lr: float = DefaultVal(0.000003), 
maxsteps: int = DefaultVal(500000), 
save_every: int = DefaultVal(None), 
warmup: int = DefaultVal(None), 
warmup_bert: int = DefaultVal(None), 
relu: bool = DefaultVal(False), 
nway: int = DefaultVal(2), 
use_ib_negatives: bool = DefaultVal(False), 
reranker: bool = DefaultVal(False), 
distillation_alpha: float = DefaultVal(1), 
ignore_scores: bool = DefaultVal(False), 
model_name: str = DefaultVal(None), 
query_maxlen: int = DefaultVal(32), 
attend_to_mask_tokens: bool = DefaultVal(False), 
interaction: str = DefaultVal("colbert"), 
dim: int = DefaultVal(128), 
doc_maxlen: int = DefaultVal(220), 
mask_punctuation: bool = DefaultVal(True), 
checkpoint: str = DefaultVal(None), 
triples: str = DefaultVal(None), 
collection: str = DefaultVal(None), 
queries: str = DefaultVal(None), 
index_name: str = DefaultVal(None), 
overwrite: bool = DefaultVal(False), 
root: str = DefaultVal(os.path.join(os.getcwd(), "experiments")), 
experiment: str = DefaultVal("default"), 
index_root: str = DefaultVal(None), 
name: str = DefaultVal(timestamp(daydir=True)), 
rank: int = DefaultVal(0), 
nranks: int = DefaultVal(1), 
amp: bool = DefaultVal(True), 
gpus: int = DefaultVal(total_visible_gpus), 
avoid_fork_if_possible: bool = DefaultVal(False)) -> ColBERTConfig
