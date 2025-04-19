CUDA_VISIBLE_DEVICES=2 python create_phrase_vocab_from_s2orc.py \
--num_slices 8 \
--current_slice_index 7 \
--output_folder /scratch/lamdo/s2orc_phrase_vocab \
--top_k_candidates 5