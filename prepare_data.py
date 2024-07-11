import os
import subprocess

def get_input(prompt, default):
    return input(f"{prompt} (default: {default}): ") or default

def run_subprocess(script, *args):
    subprocess.run(["python", os.path.join(this_dir, script), *args], check=True)

def run_scale_pyramid(zarr_file, dataset_name):
    scales = get_input("Enter scales", "1 2 2 1 2 2").split()
    chunks = get_input("Enter chunk size", "8 256 256").split()
    run_subprocess('data/scale_pyramid.py', '-f', zarr_file, '-d', dataset_name, '-s', *scales, '-c', *chunks)

def process_dataset(dataset_type, zarr_file, resolution):
    path = get_input(f"Enter path to {dataset_type} tifs / tif stack", None)
    if not path:
        return None, False
    
    dataset_name = get_input(f"Enter output {dataset_type} dataset name", f"volumes/{dataset_type}/s0")
    run_subprocess('data/make_3d_zarr_array.py', path, zarr_file, dataset_name, *resolution)
    
    ds_scale_pyramid = get_input(f"Run downscale pyramid to {dataset_type} volume?", "y").lower().strip() == 'y'
    if ds_scale_pyramid:
        run_scale_pyramid(zarr_file, dataset_name)
    
    return dataset_name, ds_scale_pyramid

def process_masks(zarr_file, dataset, mask_type):
    if not dataset:
        default_name = f"volumes/{'raw/image' if mask_type == 'raw' else 'labels/ids'}/s0"
        dataset = get_input(f"Provide {'raw image' if mask_type == 'raw' else 'labels'} dataset name", default_name), False
    
    mask_dataset = dataset[0].replace('image' if mask_type == 'raw' else 'ids', 'mask')
    source_dataset = dataset[0].replace('s0', 's2') if dataset[1] and mask_type == 'raw' else dataset[0]
    
    script = 'data/make_raw_mask.py' if mask_type == 'raw' else 'data/make_object_mask.py'
    run_subprocess(script, zarr_file, source_dataset, mask_dataset)
    
    if dataset[1]:
        run_scale_pyramid(zarr_file, mask_dataset)

def main():
    base_dir = get_input("Enter base directory", './test')
    os.makedirs(base_dir, exist_ok=True)
    
    zarr_file = get_input("Enter output zarr container name", os.path.join(base_dir, "test.zarr"))
    resolution = [
        get_input("Enter Z resolution (in world units)", "1"),
        get_input("Enter YX resolution (in world units)", "1")
    ]

    # Process raw image dataset
    img_dataset = process_dataset("raw/image", zarr_file, resolution)
    if get_input("Make raw masks?", "y").lower().strip() == 'y':
        process_masks(zarr_file, img_dataset, 'raw')

    # Process labels dataset
    labels_dataset = process_dataset("labels/ids", zarr_file, resolution)
    if get_input("Make object masks?", "y").lower().strip() == 'y':
        process_masks(zarr_file, labels_dataset, 'object')

if __name__ == "__main__":
    this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    main()
