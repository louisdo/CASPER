python -m data_processing.venue_grouping.collect_venues \
        --shard-dir /scratch/s2orc-20251016 \
        --output /scratch/lamdo/CSpR/venue_grouping/venue_index.json \
        --num-workers 16 --limit 1