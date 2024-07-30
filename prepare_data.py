import os
import subprocess
import zarr

def get_input(prompt, default):
    return input(f"\n{prompt} (default: {default}): \n") or default

def run_subprocess(script, *args):
    subprocess.run(["python", os.path.join(this_dir, script), *args], check=True)

def run_bounding_box(zarr_file, dataset_name, out_f=None, out_ds=None):
    run_subprocess('data/bounding_box_crop.py', zarr_file, dataset_name, out_f, out_ds)

def run_scale_pyramid(zarr_file, dataset_name, mode='down'):
    scales = get_input("Enter scales", "1 2 2 1 2 2").split()
    chunks = get_input("Enter chunk size", "8 256 256").split()
    run_subprocess('data/scale_pyramid.py', '-f', zarr_file, '-d', dataset_name, '-s', *scales, '-c', *chunks, '-m', mode)

def process_dataset(dataset_type, zarr_file, resolution):
    path = get_input(f"Enter path to {dataset_type} tif directory, tif stack, or zarr container", None)
    if not path:
        return None
    
    if path.endswith('.zarr'):
        in_ds = get_input(f"Enter input {dataset_type} dataset name contained in {path}", dataset_type.split('/')[0])
        dataset_name = get_input(f"Enter output {dataset_type} dataset name", in_ds)
        
        if get_input(f"\nPerform bounding box crop?", 'n').lower().strip() == 'y':
            run_bounding_box(path, in_ds, zarr_file, dataset_name)
        else:
            in_f = zarr.open(path)
            out_f = zarr.open(zarr_file, "a")
            if os.path.abspath(path) == os.path.abspath(zarr_file) and in_ds != dataset_name:
                print(f"Renaming {in_ds} to {dataset_name}..")
                in_f.store.rename(in_ds, in_ds+'__tmp')
                in_f.create_group('/'.join(dataset_name.split('/')[:-1]))
                in_f.store.rename(in_ds+'__tmp', dataset_name)
            else:
                print(f"Copying {path}/{in_ds} to {zarr_file}/{dataset_name}..")
                out_f[dataset_name] = in_f[in_ds]
    else:
        dataset_name = get_input(f"Enter output {dataset_type} dataset name", f"volumes/{dataset_type}/s0")
        run_subprocess('data/make_3d_zarr_array.py', path, zarr_file, dataset_name, *resolution)
    
    return dataset_name

def process_masks(zarr_file, dataset, mask_type):
    source_dataset = dataset
    out_mask_dataset = dataset.replace('image' if mask_type == 'img' else 'ids', 'mask')
    script = 'data/mask_blockwise.py'
    
    if get_input("Do masking downscaled dataset then upscale?", 'y').lower().strip() == 'y':
        source_dataset = source_dataset.replace('s0', 's2')
        out_mask_dataset = out_mask_dataset.replace('s0', 's2')
        run_subprocess(script, '-f', zarr_file, '-i', source_dataset, '-o', out_mask_dataset, '-m', mask_type)
        run_scale_pyramid(zarr_file, out_mask_dataset, mode='up')
    else:
        run_subprocess(script, '-f', zarr_file, '-i', source_dataset, '-o', out_mask_dataset, '-m', mask_type)

def prepare(dataset_type, zarr_file, resolution):
    dataset = process_dataset(dataset_type, zarr_file, resolution)
    
    if dataset:
        if get_input(f"Run downscale pyramid on {dataset}?", "y").lower().strip() == 'y':
            run_scale_pyramid(zarr_file, dataset)
        
        mask_type = 'img' if 'image' in dataset_type else 'obj'
        if get_input(f"Make {'raw' if mask_type == 'img' else 'object'} masks?", "y").lower().strip() == 'y':
            process_masks(zarr_file, dataset, mask_type)

def main():
    base_dir = get_input("Enter base directory", 'test')
    os.makedirs(base_dir, exist_ok=True)
    
    zarr_file = get_input("Enter output zarr container name", os.path.join(base_dir, "test.zarr"))
    resolution = [
        get_input("Enter Z resolution (in world units)", "1"),
        get_input("Enter YX resolution (in world units)", "1")
    ]

    prepare("raw/image", zarr_file, resolution)
    prepare("labels/ids", zarr_file, resolution)

if __name__ == "__main__":
    this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    main()
