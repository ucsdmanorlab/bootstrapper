import click

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
