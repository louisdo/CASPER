source ~/dungnguyen/miniconda3/bin/activate bio_extractor_llm

# python citation_context/s2orc/process_dataset_v2.py \
# --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata \
# --output_file  /scratch/academic_online/s2orc/v2/triplets_intermediate.citation_context.tsv \
# --metadata_file /scratch/academic_online/s2orc/metadata_from_api/metadata_from_api.3.jsonl \
# --special_token '<hier-concept>' > log/log.citation_context


# python cocit/s2orc/process_dataset_v2.py \
# --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata \
# --output_file /scratch/academic_online/s2orc/v2/triplets_intermediate.cocit.tsv \
# --metadata_file /scratch/academic_online/s2orc/metadata_from_api/metadata_from_api.3.jsonl \
# --special_token '<hier-concept>' > log/log.cocit


# python user_interaction/scirepeval_search/prepare_training_dataset_v2.py \
# --output_file /scratch/academic_online/s2orc/v2/triplets_intermediate.query.tsv \
# --special_token '<hier-concept>' > log/log.query


python kp/kp_datasets_v2.py \
--output_file  /scratch/academic_online/s2orc/v2/triplets_intermediate.kp.tsv \
--max_collections 100000 > log/log.kp


# python title/s2orc/process_dataset_v2.py \
# --input_folder /scratch/lamdo/s2orc/processed/extracted_metadata \
# --output_file /scratch/academic_online/s2orc/v2/triplets_intermediate.title.tsv \
# --metadata_file /scratch/academic_online/s2orc/metadata_from_api/metadata_from_api.3.jsonl \
# --special_token '<hier-concept>' > log/log.title