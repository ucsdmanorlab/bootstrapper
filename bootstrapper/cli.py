import click
import toml

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

from .styles import cli_echo


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
            "run",
        ]

    def get_command(self, ctx, cmd_name):
        ret = click.Group.get_command(self, ctx, cmd_name)
        if ret is not None:
            return ret

        aliases = {
            "prep": "prepare",
            "pred": "predict",
            "infer": "predict",
            "seg": "segment",
            "eval": "evaluate",
            "refine": "filter",
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
        config = toml.load(f)

    # determine command to run
    if "samples" in config:
        cli_echo(f"Running train command on {config_path}", "train")
        ctx.invoke(train, config_file=config_path)
    elif all(["chain_str" in config[setup] for setup in config]):
        cli_echo(f"Running predict command on {config_path}", "predict")
        ctx.invoke(predict, config_file=config_path)
    elif "affs_dataset" in config:
        cli_echo(f"Running segment command on {config_path}", "segment")
        ctx.invoke(segment, config_file=config_path)
    elif "out_result_dir" in config or "self" in config or "gt" in config:
        cli_echo(f"Running evaluate command on {config_path}", "evaluate")
        ctx.invoke(evaluate, config_file=config_path)
    elif (
        "eval_dir" in config
        or "seg_dataset_prefix" in config
        or "seg_datasets" in config
    ):
        cli_echo(f"Running filter command on {config_path}", "filter")
        ctx.invoke(filter, config_file=config_path)
    else:
        raise ValueError(f"Unable to determine command for {config_path}")
