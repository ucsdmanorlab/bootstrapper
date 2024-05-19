yaml_file=$1 # /path/to/yaml/config

# Run blockwise segmentation
python blockwise/02_extract_fragments_blockwise.py $yaml_file
python blockwise/03_agglomerate_blockwise.py $yaml_file
python blockwise/04_find_segments.py $yaml_file
python blockwise/05_extract_segments_from_lut.py $yaml_file
