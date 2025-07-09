````markdown
# Keyphrase-Generation-Evaluation

To run the main keyphrase evaluation script, execute the following command:

```bash
mkdir _gitig_samples
mkdir _gitig_results
bash keyphrase_generation/bash_scripts/kpeval_evaluation.sh
````

-----

## Keyphrase Evaluation Workflow

The keyphrase evaluation process involves two main steps: **formatting the keyphrase generation output** and then **computing various evaluation metrics** using Kpeval.

### Step 1: Format Keyphrase Generation Output for Kpeval

This step converts the raw results from your keyphrase generation pipeline into a format suitable for Kpeval's input.

```bash
input_file="/scratch/lvnguyen/keyphrase_generation_results/results_ongoing/$file" \
output_dir="_gitig_samples/" \
top_k=5 \ # This parameter truncates to top_k present keyphrases and top_k absent keyphrases.
python metrics/convert_splade_file.py
```

**Output from Step 1:** The formatted file will be saved as `_gitig_samples/_all_keyphrase_$file`.

### Step 2: Compute Kpeval Metrics

After formatting the input, this step computes various keyphrase evaluation metrics based on the Kpeval configuration.

```bash
python utils/phrase_splade_evaluation.py \
    --config-file kpeval/config.gin \
    --jsonl-file "_gitig_samples/_all_keyphrase_$file" \
    --metrics diversity,exact_matching,semantic_matching \
    --log-file-prefix _gitig_results/
```

**Output File from Step 2:** The results will be logged to files prefixed with `_gitig_results/`
-----

### Available Evaluation Metrics

 You can specify these using the `--metrics` argument in Step 2.

Here is a list of all available metrics:

  * `approximate_matching`
  * `bert_score`
  * `chatgpt`
  * `diversity`
  * `exact_matching`
  * `fg`
  * `meteor`
  * `mover_score`
  * `retrieval`
  * `rouge`
  * `semantic_matching`
  * `unieval`

### Custom Score Metrics Scripts

For custom evaluation scores scripts, refer to the scripts located in the `metrics/` directory.
Examples of custom metric implementations include:

  * `metrics/diversity_metric.py`
  * `metrics/sem_matching_metric.py`


```
```
