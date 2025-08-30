CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-14kaddedphrases \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_14kaddedphrases \
--max_collection 3000000 \
--batch_size 64 \
--num_train_epochs 2 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.9 \



CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 32 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc

CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc


CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orcfreqbased \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfroms2orcfreqbased \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc


CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orcfreqbased \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orcfreqbased \
--max_collection 10000000 \
--batch_size 32 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1



CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-15kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_15kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1


CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-15kaddedphrasesfroms2orcfreqbased \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_15kaddedphrasesfroms2orcfreqbased \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1


# added words  # /scratch/lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc 
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_word_30kaddedwordsfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc


CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orc_v2 \
--max_collection 10000000 \
--batch_size 32 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.15 \
--dataset_name s2orc


CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/bert-base-uncased-phrase-60kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/bert_phrase_60kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 32 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc



CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/unArxive/keyphrase_informativeness_combined_references/triplets_hardneg/raw.tsv" \
--input_model_name lamdo/distilbert-base-uncased-phrase-20kaddedphrases_erukgds \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_20kaddedphrases_erukgds_v2 \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name erukgds






# MSMARCO
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/msmarco_splade/data/msmarco/full_collection/raw.tsv" \
--input_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfrommsmarco \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfrommsmarco \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name msmarco






CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/s2orc/cs_corpus/collections.jsonl" \
--input_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfroms2orc_cs \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc_cs \
--fp16 1




CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name distilbert-base-uncased \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.15 \
--dataset_name s2orc \
--fp16 1





# scibert
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/scibert_phrase_30kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--max_steps 70000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1

# scibert_cs
CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/s2orc/cs_corpus/collections.jsonl" \
--input_model_name lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/scibert_phrase_30kaddedphrasesfroms2orc_cs \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--max_steps 26000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc_cs \
--fp16 1

# normal bert
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/bert-base-uncased-phrase-30kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/bert_phrase_30kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--max_steps 70000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1



# distilbert 60k
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 48 \
--num_train_epochs 1 \
--max_steps 70000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1


# distilbert 5k
CUDA_VISIBLE_DEVICES=0 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name lamdo/distilbert-base-uncased-phrase-5kaddedphrasesfroms2orc \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_5kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--max_steps 70000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1


# cocondenser
CUDA_VISIBLE_DEVICES=1 python src/main.py \
--collection_path "/scratch/lamdo/scirepeval/collections.txt" \
--input_model_name "lamdo/cocondenser-phrase-30kaddedphrasesfroms2orc" \
--output_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/cocondenser_phrase_30kaddedphrasesfroms2orc \
--max_collection 10000000 \
--batch_size 64 \
--num_train_epochs 1 \
--max_steps 70000 \
--save_steps 10000 \
--learning_rate 5e-5 \
--weight_decay 0.01 \
--special_token_mask_probability 0.85 \
--dataset_name s2orc \
--fp16 1