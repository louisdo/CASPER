python sample_dataset.py \
--input_folder /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title \
--num_samples 587564 \
--output_file /scratch/lamdo/phrase_splade_datasets/combined_sampled_cc+cocit+kp1m+query+title/raw.tsv


# to create mid size cs dataset
python sample_dataset.py \
--input_folder /scratch/lamdo/phrase_splade_datasets/combined_cc_cs_fullsize+cocit_cs_fullsize+kp20k+query_cs_fullsize+title_cs_fullsize \
--num_samples 1500000 \
--output_file /scratch/lamdo/phrase_splade_datasets/combined_mid-size-for-ablation_cc_cs_fullsize+cocit_cs_fullsize+kp20k+query_cs_fullsize+title_cs_fullsize/raw.tsv



# to create mid size multi-disciplinary dataset
python sample_dataset.py \
--input_folder /scratch/lamdo/phrase_splade_datasets/combined_cc+cocit+kp1m+query+title \
--num_samples 1500000 \
--output_file /scratch/lamdo/phrase_splade_datasets/combined_mid-size-for-ablation_cc+cocit+kp1m+query+title/raw.tsv