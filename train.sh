yaml_file=$1 # /path/to/yaml/config
model_name=$2 # "2d_model", "3d_model", "custom_model"

# Extract the setup_dir value from the YAML file
setup_dir=$(yq ".train.\"$model_name\".setup_dir" "$yaml_file")

# Run the Python command in the setup_dir directory
python "$setup_dir/train.py" "$yaml_file"
