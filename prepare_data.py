import os
import subprocess

def get_input(prompt, default):
    return input(f"\n{prompt} (default: {default}): \n") or default

def run_subprocess(script, *args):
    subprocess.run(["python", os.path.join(this_dir, script), *args], check=True)

def run_scale_pyramid(zarr_file, dataset_name, mode='down'):
    scales = get_input("Enter scales", "1 2 2 1 2 2").split()
    chunks = get_input("Enter chunk size", "8 256 256").split()
    run_subprocess('data/scale_pyramid.py', '-f', zarr_file, '-d', dataset_name, '-s', *scales, '-c', *chunks, '-m', mode)

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
    if not dataset[0]:
        dataset = get_input(f"Provide {'raw image' if mask_type == 'img' else 'labels'} dataset name", None), False

        if not dataset[0]:
            return
    
    source_dataset = dataset[0]
    out_mask_dataset = dataset[0].replace('image' if mask_type == 'img' else 'ids', 'mask')
    script = 'data/mask_blockwise.py'
    
    if dataset[1] and get_input("Do masking downscaled dataset then upscale?",'y') == 'y':
        source_dataset = source_dataset.replace('s0','s2')
        out_mask_dataset = out_mask_dataset.replace('s0','s2')
        run_subprocess(script, '-f', zarr_file, '-i', source_dataset, '-o', out_mask_dataset, '-m', mask_type)
        run_scale_pyramid(zarr_file, out_mask_dataset, mode='up')
    else:
        run_subprocess(script, '-f', zarr_file, '-i', source_dataset, '-o', out_mask_dataset, '-m', mask_type)

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
        process_masks(zarr_file, img_dataset, 'img')

    # Process labels dataset
    labels_dataset = process_dataset("labels/ids", zarr_file, resolution)
    if get_input("Make object masks?", "y").lower().strip() == 'y':
        process_masks(zarr_file, labels_dataset, 'obj')

if __name__ == "__main__":
    this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    main()
