yaml_file=$1 # /path/to/yaml/config
model_name=$2 # "2d_model", "3d_model", "custom_model"

# Run blockwise prediction
python blockwise/01_predict_blockwise.py $yaml_file $model_name
