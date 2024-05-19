yaml_file=$1 # /path/to/yaml/config

# Check if compute_lsd_errors exists in config
compute_lsd_errors=$(yq eval '.processing.compute_lsd_errors' "$yaml_file")

if [ -n "$compute_lsd_errors" ]; then
    python scripts/compute_errors.py $yaml_file
fi

python scripts/filter_segmentation.py $yaml_file
