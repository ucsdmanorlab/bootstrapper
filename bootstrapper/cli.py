import click
from .run import prepare, view, train, predict, segment, evaluate, filter, auto

@click.group()
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

if __name__ == "__main__":
    cli()