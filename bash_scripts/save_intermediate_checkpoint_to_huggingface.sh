python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfrommsmarco/checkpoint-70000 \
--source_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfrommsmarco \
--huggingface_path lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfrommsmarco-mlm-70000steps \


python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_word_30kaddedwordsfroms2orc/checkpoint-70000 \
--source_model_name lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc \
--huggingface_path lamdo/distilbert-base-uncased-word-30kaddedwordsfroms2orc-mlm-70000steps \


python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orcfreqbased/checkpoint-150000 \
--source_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orcfreqbased \
--huggingface_path lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orcfreqbased-mlm-150000steps



python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_15kaddedphrasesfroms2orcfreqbased/checkpoint-70000 \
--source_model_name lamdo/distilbert-base-uncased-phrase-15kaddedphrasesfroms2orcfreqbased \
--huggingface_path lamdo/distilbert-base-uncased-phrase-15kaddedphrasesfroms2orcfreqbased-mlm-70000steps


python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_30kaddedphrasesfroms2orc_cs \
--source_model_name lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs \
--huggingface_path lamdo/distilbert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs-mlm-26000steps



python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert/checkpoint-80000 \
--source_model_name distilbert-base-uncased \
--huggingface_path lamdo/distilbert-s2orc-mlm-80000steps \



# scibert cs
python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/scibert_phrase_30kaddedphrasesfroms2orc_cs \
--source_model_name lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs \
--huggingface_path lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc_cs-mlm-26000steps


# scibert
python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/scibert_phrase_30kaddedphrasesfroms2orc \
--source_model_name lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc \
--huggingface_path lamdo/scibert-base-uncased-phrase-30kaddedphrasesfroms2orc-mlm-70000steps

# normal bert
python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/bert_phrase_30kaddedphrasesfroms2orc \
--source_model_name lamdo/bert-base-uncased-phrase-30kaddedphrasesfroms2orc \
--huggingface_path lamdo/bert-base-uncased-phrase-30kaddedphrasesfroms2orc-mlm-70000steps



python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_60kaddedphrasesfroms2orc \
--source_model_name lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orc \
--huggingface_path lamdo/distilbert-base-uncased-phrase-60kaddedphrasesfroms2orc-mlm-70000steps


python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/distilbert_phrase_5kaddedphrasesfroms2orc \
--source_model_name lamdo/distilbert-base-uncased-phrase-5kaddedphrasesfroms2orc \
--huggingface_path lamdo/distilbert-base-uncased-phrase-5kaddedphrasesfroms2orc-mlm-70000steps


/scratch/lamdo/mlm_bert_phrase/checkpoints/cocondenser_phrase_30kaddedphrasesfroms2orc

python src/save_intermediate_checkpoint_to_huggingface.py \
--intermediate_checkpoint_dir /scratch/lamdo/mlm_bert_phrase/checkpoints/cocondenser_phrase_30kaddedphrasesfroms2orc \
--source_model_name lamdo/cocondenser-phrase-30kaddedphrasesfroms2orc \
--huggingface_path lamdo/cocondenser-phrase-30kaddedphrasesfroms2orc-mlm-70000steps