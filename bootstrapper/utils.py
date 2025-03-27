import click
from bootstrapper.data.bbox import bbox
from bootstrapper.data.clahe import clahe
from bootstrapper.data.convert import convert
from bootstrapper.data.mask import mask
from bootstrapper.data.scale_pyramid import scale_pyramid
from bootstrapper.data.merge import merge


@click.group()
def utils():
    """Utility functions for volumes and segmentations"""
    pass


utils.add_command(bbox)
utils.add_command(clahe)
utils.add_command(convert)
utils.add_command(mask)
utils.add_command(scale_pyramid)
utils.add_command(merge)

@utils.command()
@click.option("--model-name", "-m", help="Name of the model to download")
@click.option("--setup-dir", "-s", help="Directory to download checkpoints to")
def download_ckpts(model_name=None, setup_dir=None):
    """Download pretrained checkpoints for all pred models"""
    from bootstrapper.configs import MODEL_NAMES, MODEL_DIR, MODEL_SHORT_NAMES, copy_model_scripts, download_checkpoints
    import os
    import json

    pred_models = [m for m in MODEL_NAMES if "_from_" in m]

    if model_name is None:
        models = pred_models
    elif model_name in pred_models:
        models = [model_name]
    elif model_name in MODEL_SHORT_NAMES.values():
        models = [k for k, v in MODEL_SHORT_NAMES.items() if v == model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    for model in models:
        if setup_dir is None:
            model_setup_dir = os.path.join(MODEL_DIR, model)
            download_checkpoints(model, model_setup_dir)
        else:
            model_setup_dir = setup_dir if len(models) == 1 else os.path.join(setup_dir, model)
            net_config_path = os.path.join(model_setup_dir, "net_config.json")
            if os.path.exists(net_config_path):
                
                with open(net_config_path) as f:
                    setup_config = json.load(f)
                
                with open(os.path.join(MODEL_DIR, model, "net_config.json")) as f:
                    model_config = json.load(f)
                    
                if setup_config != model_config:
                    raise ValueError(f"net_config.json in {model_setup_dir} does not match {model_name}")
            else:
                copy_model_scripts(model, model_setup_dir, cli_edit=False)
                download_checkpoints(model, model_setup_dir)


