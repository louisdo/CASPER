models_to_include=(
    phrase_splade_71
)

join_by() {
  local separator="$1"
  shift
  local first="$1"
  shift
  printf "%s" "$first" "${@/#/$separator}"
}

models_to_include_joined=$(join_by , "${models_to_include[@]}")

echo "$models_to_include_joined"

RESULTS_FOLDER="/scratch/lvnguyen/phrase_splade_keyphrase_generation_results" \
DATASETS_TO_INCLUDE="semeval,inspec,nus,krapivin" MODELS_TO_INCLUDE=$models_to_include_joined \
python view_kg_experiment_results.py

# "semeval,inspec,nus,krapivin,kp20k"