CUDA_VISIBLE_DEVICES=2 python create_phrase_vocab_from_msmarco.py \
--input_file /scratch/lamdo/msmarco_splade/data/msmarco/full_collection/raw.tsv \
--num_slices 4 \
--current_slice_index 3 \
--output_folder /scratch/lamdo/msmarco_phrase_vocab \
--top_k_candidates 5