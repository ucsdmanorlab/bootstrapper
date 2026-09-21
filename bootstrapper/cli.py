import glob
import os
import sys
import time
import traceback

import click

from . import (
    prepare,
    train,
    predict,
    segment,
    evaluate,
    refine,
    view,
    utils,
)

from .config import load, runs, OPS
from .refine import MORPH_OPS
from .styles import cli_echo


class CommandGroup(click.Group):
    def list_commands(self, ctx):
        return [
            "prepare",
            "train",
            "predict",
            "segment",
            "evaluate",
            "refine",
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
        }

        if cmd_name in aliases:
            return click.Group.get_command(self, ctx, aliases[cmd_name])
        return None

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.Abort, click.exceptions.Exit):
            raise
        except Exception as e:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            log = os.path.abspath(f"bs_error_{stamp}.log")
            with open(log, "w") as f:
                f.write(f"{stamp} {os.getcwd()}\n$ {' '.join(sys.argv)}\n\n")
                traceback.print_exc(file=f)
            raise click.ClickException(f"{type(e).__name__}: {e} (traceback in {log})") from e


@click.group(cls=CommandGroup)
@click.version_option(package_name="bootstrapper")
def cli():
    """Bootstrapper CLI"""
    pass


cli.add_command(prepare)
cli.add_command(view)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(segment)
cli.add_command(evaluate)
cli.add_command(refine)
cli.add_command(utils)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--only", multiple=True, help="Run only these steps: a kind, kind.name, or a file prefix in a directory")
@click.option("--from", "from_", help="Run from this step to the end")
@click.pass_context
def run(ctx, path, only, from_):
    """Run the steps of a config file in file order, or the numbered files of a directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "[0-9]*.toml")))
        if not files:
            raise click.ClickException(f"{path}: no numbered .toml files")
    else:
        files = [path]
    todo = [(f, s) for f in files for s in load(f).steps]

    def match(sel, f, s):
        return sel in (s.kind, s.label) or os.path.basename(f).startswith(sel)

    if from_:
        first = next((i for i, (f, s) in enumerate(todo) if match(from_, f, s)), None)
        if first is None:
            raise click.ClickException(f"--from {from_}: no such step; steps are " + ", ".join(s.label for _, s in todo))
        todo = todo[first:]
    if only:
        todo = [(f, s) for f, s in todo if any(match(o, f, s) for o in only)]
        if not todo:
            raise click.ClickException(f"--only {' '.join(only)}: no such step")

    for f, s in todo:
        cli_echo(f"Running {s.label} from {f}", s.kind)
        if s.kind == "refine":
            run_refine(ctx, s)
        else:
            command = {"train": train, "predict": predict, "segment": segment, "evaluate": evaluate}[s.kind]
            ctx.invoke(command, config_file=f, step=None if s.name == s.kind else s.name)


def run_refine(ctx, step):
    """Run a refine step's ops in order; each op's output feeds the next."""
    ops = step.keys.get("ops", {})
    if not ops:
        raise click.ClickException(f"[{step.label}] has no ops; ops are {', '.join(OPS)}")
    for _, config in runs(step):
        array = config.get("in_array")
        if array is None:
            raise click.ClickException(f"[{step.label}] needs in_array")
        for op, keys in ops.items():
            keys = dict(keys)
            if op.replace("-", "_") in MORPH_OPS:
                keys["op"] = op.replace("-", "_")
                op = "morph"
            array = ctx.invoke(refine.commands[op], in_array=array, **keys)


