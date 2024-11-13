import click
import yaml

from . import (
    prepare,
    train,
    predict,
    segment,
    evaluate,
    filter,
    view,
    utils,
)


class CommandGroup(click.Group):
    def list_commands(self, ctx):
        # Return the commands in the desired order
        return [
            "prepare",
            "train",
            "predict",
            "segment",
            "evaluate",
            "filter",
            "view",
            "utils",
            "run"
        ]
    
    def get_command(self, ctx, cmd_name):
        ret = click.Group.get_command(self, ctx, cmd_name)
        if ret is not None:
            return ret
        
        aliases = {
            'prep': 'prepare',
            'pred': 'predict',
            'infer': 'predict',
            'seg': 'segment',
            'eval': 'evaluate',
            'refine': 'filter',       
        }

        if cmd_name in aliases:
            return click.Group.get_command(self, ctx, aliases[cmd_name])
        return None

@click.group(cls=CommandGroup)
def cli():
    """Bootstrapper CLI"""
    pass


cli.add_command(prepare)
cli.add_command(view)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(segment)
cli.add_command(evaluate)
cli.add_command(filter)
cli.add_command(utils)

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.pass_context
def run(ctx, config_path):
    """Run the appropriate command based on the config file"""

    # load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # determine command to run
    if "samples" in config:
        click.secho(f"Running train command on {config_path}")
        ctx.invoke(train, yaml_file=config_path)
    elif "chain_str" in config[config.keys()[0]]:
        click.secho(f"Running predict command on {config_path}")
        ctx.invoke(predict, yaml_file=config_path)
    elif "affs_dataset" in config:
        click.secho(f"Running segment command on {config_path}")
        ctx.invoke(segment, yaml_file=config_path)
    elif "out_result_dir" in config or "self" in config or "gt" in config:
        click.secho(f"Running evaluate command on {config_path}")
        ctx.invoke(evaluate, yaml_file=config_path)
    elif "eval_dir" in config or "seg__dataset_prefix" in config or "seg_datasets" in config:
        click.secho(f"Running filter command on {config_path}")
        ctx.invoke(filter, config_file=config_path)
    else:
        raise ValueError(f"Unable to determine command for {config_path}")

