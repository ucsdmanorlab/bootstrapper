import click

from bootstrapper.run import prepare, view, train, predict, segment, evaluate, filter, auto


class OrderedGroup(click.Group):
    def list_commands(self, ctx):
        # Return the commands in the desired order
        return [
            "prepare",
            "view",
            "train",
            "predict",
            "segment",
            "evaluate",
            "filter",
            "auto",
        ]


@click.group(cls=OrderedGroup)
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
cli.add_command(auto)
