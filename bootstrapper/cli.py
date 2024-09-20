import click
from .run import prepare, view, train, predict, segment, evaluate, filter, auto

class OrderedGroup(click.Group):
    def list_commands(self, ctx):
        # Return the commands in the desired order
        return ['prepare', 'view', 'train', 'predict', 'segment', 'evaluate', 'filter', 'auto']


@click.group(cls=OrderedGroup)
def cli():
    """Bootstrapper CLI"""
    pass

cli.add_command(prepare.run, name='prepare')
cli.add_command(view.run, name='view')
cli.add_command(train.run, name='train')
cli.add_command(predict.run, name='predict')
cli.add_command(segment.run, name='segment')
cli.add_command(evaluate.run, name='evaluate')
cli.add_command(filter.run, name='filter')
cli.add_command(auto.run, name='auto')