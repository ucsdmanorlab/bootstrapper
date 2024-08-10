import numpy as np
import json
import zarr
import os
import sys
from shutil import copytree
import yaml
import pprint

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def get_volumes():
    volumes = []
    num_volumes = int(input("How many volumes for this round? (default = 1)") or 1)
    for i in range(num_volumes):
        print(f"Volume {i+1}:")
        vol = {}
        vol['zarr_container'] = input("Enter Zarr container path: ")
        vol['raw_dataset'] = input("Enter raw dataset path: ")
        vol['raw_mask_dataset'] = input("Enter raw mask dataset path (or leave blank if none): ") or None
        vol['labels_dataset'] = input("Enter labels dataset path (or leave blank if none): ") or None
        vol['unlabelled_mask_dataset'] = input("Enter unlabelled mask dataset path (or leave blank if none): ") or None
        with zarr.open(vol['zarr_container'], 'r') as f:
            voxel_size = f[vol['raw_dataset']].attrs['resolution']
        same_voxel_size = input(f"Voxel size for this image volume is {voxel_size}. Use this? (y/n) ") or "y"
        same_voxel_size = same_voxel_size.lower().strip() == 'y'
        if same_voxel_size:
            vol['voxel_size'] = voxel_size
        else:
            vol['voxel_size'] = list(map(int, input("Enter voxel size (comma-separated): ").split(',')))
        print('\n')
    
        volumes.append(vol)

    return volumes


def get_rag_db_config(sqlite_path=None):

    nodes_table = input("Enter RAG nodes table name (default = 'nodes'): ") or "nodes"
    edges_table = input("Enter RAG edges table name (default = 'edges'): ") or "edges"
   
    # confirm SQLite or not
    use_sqlite = True if sqlite_path is not None else False
    confirm_sqlite = input(f'use_sqlite = {use_sqlite}. Continue? (y/n): ') or "y"
    confirm_sqlite = confirm_sqlite.lower().strip() == 'y'
    use_sqlite = use_sqlite if confirm_sqlite else not use_sqlite

    if use_sqlite:
        db_file = input(f"Enter SQLite RAG database file (default: {sqlite_path}): ") or sqlite_path

        return {
            'db_file': db_file,
            'nodes_table': nodes_table,
            'edges_table': edges_table
        }

    else:
        db_host = os.environ.get('RAG_DB_HOST')
        db_user = os.environ.get('RAG_DB_USER')
        db_password = os.environ.get('RAG_DB_PASSWORD')
        db_port = os.environ.get('RAG_DB_PORT')

        if not all([db_host, db_user, db_password, db_port]):
            print("PgSQL Database credentials not found in environment variables.")
            db_host = input("Enter PgSQL RAG database host: ")
            db_user = input("Enter PgSQL RAG database user: ")
            db_password = input("Enter PgSQL RAG database password: ")
            db_port = int(input("Enter PgSQL RAG database port: "))
        
        db_name = input("Enter PgSQL RAG database name: ")
        # write to env
        os.environ['RAG_DB_HOST'] = db_host
        os.environ['RAG_DB_USER'] = db_user
        os.environ['RAG_DB_PASSWORD'] = db_password
        os.environ['RAG_DB_PORT'] = str(db_port)

        return {
            'db_host': db_host,
            'db_user': db_user,
            'db_password': db_password,
            'db_port': db_port,
            'db_name': db_name,
            'nodes_table': nodes_table,
            'edges_table': edges_table
        }


def copy_model_scripts(model_name, setup_dir):
    src = os.path.join(this_dir,'models',model_name)
    dst = setup_dir
    print(f"Copying {src} to {dst}")
    copytree(src, dst)


def check_and_update(defaults):
    pprint.pp(defaults)
    user_input = input("Are these values good? (y/n) ") or "y"
    if user_input.lower() == "n":
        params = defaults.copy()  # Create a copy to update
        while True:
            param_name = input("Enter the parameter name to change (or 'done' to finish): ")
            if param_name.lower() == "done":
                break
            new_value = input(f"Enter the new value for {param_name}: ")
            # Update the value based on the parameter name
            if param_name in params:
                param_type = type(params[param_name])
                if param_type is bool:
                    params[param_name] = new_value.lower() == "true"
                elif param_type is float:
                    params[param_name] = float(new_value)
                elif param_type is int:
                    params[param_name] = int(new_value)
                elif param_type is list:
                    # Convert the input string to a list, then convert each element to the appropriate type
                    new_value_list = eval(new_value)
                    if len(params[param_name]) > 0:
                        element_type = type(params[param_name][0])
                        params[param_name] = [element_type(x.strip()) for x in new_value_list]
                    else:
                        params[param_name] = [x.strip() for x in new_value_list]
                elif param_type is str:
                    params[param_name] = new_value if new_value.lower() != "none" else None
                else:
                    print(f"Unsupported data type for {param_name}: {param_type}")
            else:
                print(f"Invalid parameter name: {param_name}")
        # Print updated values
        print("\nUpdated values:")
        pprint.pp(params)
        return params
    else:
        return defaults

def make_round_configs(base_dir, round_number, round_name=None, previous_round=None): 

    i = round_number
    
    # previous round
    if previous_round is not None:
        previous_round_name = list(revious_round.keys())[0]
    else:
        if i != 0:
            previous_round_name = f"round_{i-1}"

    # get volumes for this round
    volumes = get_volumes()

    # round number, name
    voxel_size = volumes[0]['voxel_size']  # Assuming voxel_size is the same for all volumes
    if round_name is None:
        round_name = f"round_{i}"

    # training config
    model_name = input(f"Enter model for {round_name}: ") or ('2d_mtlsd' if i == 0 else '3d_lsd')
    setup_dir = os.path.abspath(os.path.join(base_dir, f'{round_name}', model_name))
    copy_model_scripts(model_name, setup_dir)
    
    if 'lsd' in model_name.lower():
        default_sigma = 10 * voxel_size[-1]
        sigma = int(input(f"Enter sigma for LSD model (default = {default_sigma}): ") or default_sigma)
    else:
        sigma = default_sigma
    
    max_iterations = int(input(f"Enter max iterations for {round_name} training (default is 30001): ") or 30001) 
    save_checkpoints_every = int(input(f"Enter save checkpoints every for {round_name} (default is 5000): ") or 5000)
    save_snapshots_every = int(input(f"Enter save snapshots every for {round_name} (default is 1000): ") or 1000)

    training_samples = [v for v in volumes if None not in [v['labels_dataset'],v['unlabelled_mask_dataset']]]
    train_config = check_and_update({
            'setup_dir': setup_dir,
            'samples': [x['zarr_container'] for x in training_samples],
            'raw_datasets': [x['raw_dataset'] for x in training_samples],
            'labels_datasets': [x['labels_dataset'] for x in training_samples],
            'mask_datasets': [x['unlabelled_mask_dataset'] for x in training_samples],
            'voxel_size': voxel_size,
            'sigma': sigma,
            'out_dir': setup_dir,
            'max_iterations': max_iterations,
            'save_checkpoints_every': save_checkpoints_every,
            'save_snapshots_every': save_snapshots_every,
    })
    print('\n')

    # predict, segment, filter configs
    print("Predict, segment, and filter configs for all target volumes")
    pred_iter = int(input(f"Enter checkpoint iteration for {round_name} inference (default is {max_iterations - 1}): ") or max_iterations - 1)

    # get model prediction outputs
    with open(os.path.join(setup_dir,'config.json'),'r') as f:
        model_outputs = json.load(f)['outputs']

    pred_datasets = [f'predictions/{round_name}/{x}_{pred_iter}' for x in model_outputs] 

    # are 3d affs needed ?
    make_affs = True if '3d_affs' not in model_outputs else False 
    if make_affs:
        affs_setup_dir = os.path.join(this_dir,'models/2d_mtlsd_to_affs') \
                if ('2d_lsds' in model_outputs and '2d_affs' in model_outputs) \
                else os.path.join(this_dir,'models/3d_lsd_to_affs') 
        affs_iter = int(input(f"Enter checkpoint iteration for round {i} inference of affs with {affs_setup_dir} (default is 2000): ") or 2000) 
        out_affs_ds = f"predictions/{round_name}/3d_affs_{affs_iter}_from_{pred_datasets[0].split('_')[-1]}" 
    else:
        out_affs_ds = [x for x in pred_datasets if '3d_affs' in x][0]
   
    # can lsd errors be computed ?
    compute_lsd_errors = True if '3d_lsds' in model_outputs else False

    # segmentation parameters
    seg_params = {
        "fragments_in_xy" : True,
        "background_mask" : False,
        "mask_thresh" : 0.5,
        "min_seed_distance" : 10,
        "epsilon_agglomerate" : 0.0,
        "filter_fragments" : 0.05,
        "replace_sections" : None,
        "mask_file" : None,
        "mask_dataset" : None,
        "roi_offset" : None,
        "roi_shape" : None,
        "block_size" : None,
        "context" : None,
        "num_workers" : 20,
        "thresholds_minmax" : [0, 1],
        "thresholds_step" : 0.05,
        "thresholds": [0.2, 0.3, 0.4, 0.5],
        "merge_function" : "mean",
        "merge_threshold" : 30,
    }

    # filter parameters
    if compute_lsd_errors:
        lsd_error_thresholds = [0.1,1.0]
        use_lsd_errors = input('Use LSD Errors mask to filter segmentation? (y/n): ') or "y"
        use_lsd_errors = use_lsd_errors.lower().strip() == 'y'
    else:
        use_lsd_errors = False

    filter_params = {
        "dust_filter" : 1000,
        "remove_outliers" : True,
        "remove_z_fragments" : 10,
        "overlap_filter": 0.0,
        "erode_out_mask" : False,
    }

    # target volume configs
    target_volumes = volumes
    target_configs = {}
    for t_vol in target_volumes:

        # get volume shape
        full_shape = zarr.open(t_vol['zarr_container'],"r")[t_vol['raw_dataset']].shape

        # get ROI
        print('\n')
        print(f"{t_vol['zarr_container']} target ROIs -- comma-separated world units, not voxels, default is None: ")
        roi_offset = list(map(int, input("Enter ROI offset: ").split(','))) if input("Enter ROI offset: ") else None
        roi_shape = list(map(int, input("Enter ROI shape: ").split(','))) if input("Enter ROI shape: ") else None
        
        if roi_offset is None:
            roi_offset = zarr.open(t_vol['zarr_container'],"r")[t_vol['raw_dataset']].attrs['offset']

        if roi_shape is not None:
            shape = [x/y for x,y in zip(roi_shape,voxel_size)]
        else:
            shape = full_shape
            roi_shape = [x * y for x,y in zip(shape,voxel_size)]

        # small volume or not ?
        small_volume = False
        if np.prod(shape) <= 134217728: # 512x512x512 uint64, ~1GB
            small_volume = True

        # inference
        pred_config = check_and_update({
            'setup_dir': setup_dir,
            'raw_file': t_vol['zarr_container'],
            'raw_datasets': [t_vol['raw_dataset'],],
            'roi_offset': roi_offset,
            'roi_shape': roi_shape,
            'checkpoint': os.path.join(setup_dir,f"model_checkpoint_{pred_iter}"),
            'out_file': t_vol['zarr_container'],
            'out_prefix': f"predictions/{round_name}",
            'num_workers': 6,
            'num_gpus': 3,
            'num_cache_workers': 1
        })

        # 3d affs inference, if needed
        if make_affs:
            pred_affs_config = check_and_update({
                'setup_dir': affs_setup_dir,
                'raw_file': t_vol['zarr_container'],
                'raw_datasets': pred_datasets,
                'roi_offset': roi_offset,
                'roi_shape': roi_shape,
                'checkpoint': os.path.join(affs_setup_dir,f"model_checkpoint_{affs_iter}"),
                'out_file': t_vol['zarr_container'],
                'out_prefix': f"predictions/{round_name}",
                'num_workers': 6,
                'num_gpus': 3,
                'num_cache_workers': 1
            })
        else:
            pred_affs_config = {}
        
        # get db config
        print(f"{round_name} database config for {t_vol['zarr_container']}:")
        sqlite_path = os.path.join(t_vol['zarr_container'],f'post/{round_name}/rag.db') if small_volume else None
        db_config = check_and_update(get_rag_db_config(sqlite_path=sqlite_path)) 

        # blockwise or not ?
        do_blockwise = True
        if small_volume:
            do_blockwise = False
        confirm_blockwise = input(f'do_blockwise = {do_blockwise}. Continue? (y/n): ') or "y"
        confirm_blockwise = confirm_blockwise.lower().strip() == 'y'
        do_blockwise = do_blockwise if confirm_blockwise else not do_blockwise

        if not do_blockwise:
            seg_params['num_workers'] = 1
            seg_params['block_size'] = roi_shape 

        seg_params['roi_offset'] = roi_offset
        seg_params['roi_shape'] = roi_shape

        # use raw mask in processing ?
        use_raw_mask = t_vol['raw_mask_dataset'] is not None
        if use_raw_mask:
            confirm_raw_mask = input(f'use_raw_mask = {use_raw_mask}. Continue? (y/n): ') or "y"
            confirm_raw_mask = confirm_raw_mask.lower().strip() == 'y'
            use_raw_mask = use_raw_mask if confirm_raw_mask else not use_raw_mask
            seg_params['mask_file'] = t_vol['zarr_container']
            seg_params['mask_dataset'] = t_vol['raw_mask_dataset']

        # segmentation
        seg_config = check_and_update({
            'affs_file': t_vol['zarr_container'],
            'affs_dataset': out_affs_ds,
            'fragments_file': t_vol['zarr_container'],
            'fragments_dataset': f'post/{round_name}/fragments',
            'lut_dir': f"post/{round_name}/luts",
            'seg_file': t_vol['zarr_container'],
            'seg_dataset': f'post/{round_name}/segmentations', 
        } | seg_params)


        # self-evaluation
        if compute_lsd_errors == True:
            lsd_error_config = check_and_update({
                'seg_file': t_vol['zarr_container'],
                'seg_dataset': f"post/{round_name}/segmentations/{seg_params['merge_function']}/{seg_params['merge_threshold']}",
                'lsds_file': t_vol['zarr_container'],
                'lsds_dataset': f"predictions/{round_name}/3d_lsds_{pred_iter}",
                'mask_file': None,
                'mask_dataset': None,
                'out_file': t_vol['zarr_container'],
                'out_map_dataset': f"post/{round_name}/lsd_errors/map",
                'out_mask_dataset': f"post/{round_name}/lsd_errors/mask",
                'thresholds': lsd_error_thresholds
            })

        else:
            lsd_error_config = {}
    
        # filtering and pseudo ground-truth
        filter_config = check_and_update({
            'seg_file': t_vol['zarr_container'],
            'seg_dataset': f"post/{round_name}/segmentations/{seg_params['merge_function']}/{seg_params['merge_threshold']}",
            'out_file': t_vol['zarr_container'],
            'out_labels_dataset': f"pseudo_gt/{round_name}/ids",
            'out_mask_dataset': f"pseudo_gt/{round_name}/mask",
            'lsd_error_file': lsd_error_config['out_file'] if use_lsd_errors else None,
            'lsd_error_mask_dataset': lsd_error_config['out_mask_dataset'] if use_lsd_errors else None,
        } | filter_params)

        target_configs[t_vol['zarr_container']] = {
            'predict': {
                model_name: pred_config, 
                'affs': pred_affs_config
            },
            'db': db_config,
            'processing': {
                'hglom_segment': seg_config,
                'compute_lsd_errors': lsd_error_config,
                'filter': filter_config
            }
        }

    # make round config
    rnd = {}
    rnd['volumes'] = volumes
    rnd[f'train_{model_name}'] = {'train': train_config}
    for target, config in target_configs.items():
        rnd[os.path.basename(target).split('.')[0]] = config

    # write round configs
    rc_dir = os.path.join(base_dir,'configs',round_name)
    os.makedirs(rc_dir, exist_ok=True)

    for step in rnd:
        out_config_file = os.path.join(rc_dir,f'{step}.yaml')
        print(f"Writing {out_config_file}..")

        with open(out_config_file, 'a') as f:
            yaml.dump(rnd[step], f)            

    print(f"All configuration files for {round_name} generated successfully!")


def main():
    base_dir = input("Enter base directory: ") or "./test"
    config_dir = os.path.join(base_dir,'configs')
    
    if os.path.exists(config_dir) and len(os.listdir(config_dir)) > 0:
        existing_rounds = os.listdir(config_dir)

        print(f"Existing rounds at {config_dir}: {existing_rounds}")

        round_number = int(input(f"Enter new round number (default: {len(existing_rounds)}): ") or len(existing_rounds))
        round_name = input(f"Enter name for round number {round_number} (default: 'round_{round_number}'):") or None
        previous_round = input(f"Base this round off a previous round? If yes, enter round name:") or None

        if previous_round is not None:
            prc = {}
            for yc in glob.glob(os.path.join(config_dir,previous_round,'*')):
                step = os.path.basename(yc).split('.')[0]
                with open(yc, 'r') as f:
                    prc[step] = yaml.safe_load(f)
        else:
            prc = None

        print(f"Preparing configs for round number {round_number}..")
        rc = make_round_configs(base_dir, round_number, round_name=round_name, previous_round=prc)

    else:
        os.makedirs(config_dir, exist_ok = True)
        print(f"Preparing configs for round_0..")
        rc = make_round_configs(base_dir, 0, round_name=None, previous_round=None)


if __name__ == "__main__":
    main()
