import click
import os
import yaml

#from ..utils import get_volumes, make_configs


@click.group()
@click.pass_context
def run(ctx):
    """Prepare volumes and configurations"""
    ctx.ensure_object(dict)
    base_dir = click.prompt("Enter the base directory path", default=".", type=click.Path(file_okay=False, dir_okay=True))
    os.makedirs(base_dir, exist_ok=True)
    ctx.obj['base_dir'] = base_dir
    ctx.obj['volumes'] = []


@run.command(name="data")
@click.pass_context
def data(ctx):
    """Prepare volumes"""
    click.echo(f"Preparing volumes in {ctx.obj['base_dir']}...")
    ctx.obj['volumes'] = get_volumes(ctx.obj['base_dir'])

    # Dump volumes to a yaml in the base directory
    volumes_yaml = os.path.join(ctx.obj['base_dir'], 'volumes.yaml')
    with open(volumes_yaml, 'w') as f:
        yaml.dump(ctx.obj['volumes'], f)
    click.echo(f"Volumes saved to {volumes_yaml}")


@run.command(name="configs")
@click.pass_context
def configs(ctx):
    """Prepare configuration files"""
    click.echo(f"Preparing configuration files in {ctx.obj['base_dir']}...")

    # get volumes from yaml if not already in context
    # run data command if volumes yaml does not exist
    volumes_yaml = os.path.join(ctx.obj['base_dir'], 'volumes.yaml')
    if not ctx.obj['volumes']:
        if os.path.exists(volumes_yaml):
            with open(volumes_yaml, 'r') as f:
                ctx.obj['volumes'] = yaml.load(f, Loader=yaml.FullLoader)
        else:
            click.echo(f"Volumes not found. Running data command...")
            ctx.invoke(data)

    make_configs(ctx.obj['volumes'], ctx.obj['base_dir'])
    