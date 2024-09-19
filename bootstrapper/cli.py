import click
import subprocess
import os

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def run_subprocess(script, yaml_file=None, **kwargs):
    cmd = ["python", os.path.join(this_dir, script)]
    if yaml_file:
        cmd.extend(["--yaml_file", yaml_file])
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    subprocess.run(cmd, check=True)

@click.group()
def cli():
    """Bootstrapper CLI"""
    pass

def create_command(name, script, help, needs_yaml=False, allow_kwargs=False, extra_params=None):
    params = []
    if needs_yaml:
        params.append(click.option('--yaml_file', type=click.Path(exists=True), required=True, help="Path to the YAML configuration file"))
    if extra_params:
        params.extend(extra_params)
    if allow_kwargs:
        params.append(click.argument('kwargs', nargs=-1, type=click.UNPROCESSED))

    @cli.command(name, help=help)
    @click.pass_context
    def command(ctx, **kwargs):
        yaml_file = kwargs.pop('yaml_file', None)
        extra_kwargs = kwargs.pop('kwargs', None)
        if extra_kwargs:
            kwargs.update(dict(arg.split('=') for arg in extra_kwargs))
        run_subprocess(script, yaml_file, **kwargs)

    for param in params:
        command = param(command)
    return command

# Define commands
commands = [
    {
        "name": "prepare",
        "script": "configs.py",
        "help": "Prepare configurations",
        "allow_kwargs": True
    },
    {
        "name": "convert",
        "script": "convert.py",
        "help": "Run conversion",
        "allow_kwargs": True
    },
    {
        "name": "train",
        "script": "train.py",
        "help": "Run training",
        "needs_yaml": True,
        "allow_kwargs": True
    },
    {
        "name": "predict",
        "script": "predict.py",
        "help": "Run prediction",
        "needs_yaml": True,
        "extra_params": [click.option('--model', required=True, help="Name of the model")]
    },
    {
        "name": "segment",
        "script": "segment.py",
        "help": "Run segmentation",
        "needs_yaml": True,
        "allow_kwargs": True
    },
    {
        "name": "eval",
        "script": "post/compute_errors.py",
        "help": "Compute errors and evaluate results",
        "needs_yaml": True,
        "allow_kwargs": True
    },
    {
        "name": "filter",
        "script": "post/filter_segmentation.py",
        "help": "Run filtering",
        "needs_yaml": True,
        "allow_kwargs": True
    },
]

# Create commands
for cmd in commands:
    create_command(**cmd)

def main():
    cli()

if __name__ == "__main__":
    main()