import os
import sys
import zarr
import numpy as np
import subprocess
import ast

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def main():
    # base dir ? make if DNE
    base_dir = input("Enter base directory (default:'./exp_dir'): ") or './exp_dir'
    os.makedirs(base_dir, exist_ok=True) 

    # out zarr ?  
    out_zarr_f = input(f"Enter output zarr container name (default: '{os.path.join(base_dir,test.zarr')}): ") or os.path.join(base_dir,"test.zarr")

    # output resolution ?
    out_z_res = int(input("Enter Z resolution (in world units, default = 1):") or 1)
    out_yx_res = int(input("Enter YX resolution (in world units, default = 1):") or 1)
    out_vs = [out_z_res, out_yx_res, out_yx_res]
    
    # image dataset ?
    img_path = input("Enter path to aligned image tif stack or directory containing aligned image tifs (Enter to skip): ") or None
    if img_path:
        out_img_ds_name = input("Enter output image dataset name (default: 'volumes/image/s0'):") or "volumes/image/s0"

        # run img(s) to zarr
        subprocess.run(["python",os.path.join(this_dir,'data/make_3d_zarr_array.py'),img_path,out_zarr_f,out_img_ds_name,out_z_res,out_yx_res], check=True)

        # run scale pyramid ?
        img_down_scale_pyr = input("Run downscale pyramid to image volume? (y/n)") or "y" 
        img_down_scale_pyr = img_down_scale_pyr.lower().strip() == 'y'

        if img_down_scale_pyr:
            scales = input("Enter scales (default = [[1,2,2],[1,2,2]]):")
            scales = ast.literal_eval(scales) if scales else [[1,2,2],[1,2,2]]
            chunks = input("Enter chunk size (default = [8,256,256]):") 
            chunks = ast.literal_eval(chunks) if chunks else [8,256,256]

            subprocess.run(["python",os.path.join(this_dir,'data/scale_pyramid.py'),'-f',out_zarr_f,'-d',out_img_ds_name,'-s',scales,'-c',chunks],check=True)

        # make raw mask
        make_raw_mask = input("Make raw masks? (y/n): ") or "y"
        make_raw_mask = make_raw_mask.lower().strip() == 'y'

        if make_raw_mask:
            out_raw_mask_ds_name = out_img_ds_name.replace('image','image_mask')
            subprocess.run(["python",os.path.join(this_dir,'data/make_raw_mask.py'),out_zarr_f,out_img_ds_name,out_raw_mask_ds_name],check=True)
            
            if img_down_scale_pyr:
                subprocess.run(["python",os.path.join(this_dir,'data/scale_pyramid.py'),'-f',out_zarr_f,'-d',out_raw_mask_ds_name,'-s',scales,'-c',chunks],check=True)


    # labels dataset ?
    labels_path = input("Enter path to labels tifs / tif stack (Enter to skip): ") or None
    if labels_path:
        out_labels_ds_name = input("Enter output labels dataset name (default: 'volumes/labels/ids/s0'):") or "volumes/labels/ids/s0"

        # run labels to zarr
        subprocess.run(["python",os.path.join(this_dir,'data/make_3d_zarr_array.py'),labels_path,out_zarr_f,out_labels_ds_name,out_z_res,out_yx_res], check=True)

        # run scale pyramid ?
        labels_down_scale_pyr = input("Run downscale pyramid to labels volume? (y/n)") or "y" 
        labels_down_scale_pyr = labels_down_scale_pyr.lower().strip() == 'y'

        if labels_down_scale_pyr:
            scales = input("Enter scales (default = [[1,2,2],[1,2,2]]):")
            scales = ast.literal_eval(scales) if scales else [[1,2,2],[1,2,2]]
            chunks = input("Enter chunk size (default = [8,256,256]):") 
            chunks = ast.literal_eval(chunks) if chunks else [8,256,256]

            subprocess.run(["python",os.path.join(this_dir,'data/scale_pyramid.py'),'-f',out_zarr_f,'-d',out_labels_ds_name,'-s',scales,'-c',chunks],check=True)

        # make unlabelled mask 
        make_obj_mask = input("Make object masks? (y/n): ") or "y"
        make_obj_mask = make_object_mask.lower().strip() == 'y'

        if make_obj_mask:
            out_obj_mask_ds_name = out_labels_ds_name.replace('ids','mask')
            subprocess.run(["python",os.path.join(this_dir,'data/make_object_mask.py'),out_zarr_f,out_labels_ds_name,out_obj_mask_ds_name],check=True)
            
            if labels_down_scale_pyr:
                subprocess.run(["python",os.path.join(this_dir,'data/scale_pyramid.py'),'-f',out_zarr_f,'-d',out_obj_mask_ds_name,'-s',scales,'-c',chunks],check=True)


if __name__ == "__main__":
    main()
