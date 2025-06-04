CUDA_VISIBLE_DEVICES=1 python create_phrase_vocab_from_s2orc_cs.py \
--path /scratch/lamdo/s2orc/cs_corpus/collections.jsonl \
--num_slices 4 \
--current_slice_index 3 \
--output_folder /scratch/lamdo/s2orc_cs_phrase_vocab \
--top_k_candidates 5