import os
import platform
import re
import sys
import time
import traceback
from importlib.metadata import version

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

from .config import READS, WRITES, OPS, load_volumes, matches, numbered, ops_beside, paths, prepared_round, refuse_unknown, runs
from .configs import MODEL_DIR
from .prepare import stale
from .refine import MORPH_OPS
from .styles import cli_echo
from .write import file_text


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
@click.option("--dry-run", is_flag=True, help="Check every path and print what each step reads and writes; run nothing")
@click.pass_context
def run(ctx, path, only, from_, dry_run):
    """Run the steps of a config file in file order, or the numbered files of a directory. Each
    run writes a receipt into runs/: the keys each step read, which bs run repeats."""
    todo = numbered(path)
    if from_:
        first = next((i for i, step in enumerate(todo) if matches(from_, *step)), None)
        if first is None:
            raise click.ClickException(f"--from {from_}: no such step; steps are " + ", ".join(s.label for _, _, s in todo))
        todo = todo[first:]
    if only:
        todo = [step for step in todo if any(matches(o, *step) for o in only)]
        if not todo:
            raise click.ClickException(f"--only {' '.join(only)}: no such step")

    refuse_stale(path)
    volumes = load_volumes(todo[0][1]) if todo else {}
    lines = plan(path, todo, volumes)
    if dry_run:
        click.echo("\n".join(lines))
        return

    root = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
    receipt = os.path.join(root, "" if os.path.basename(root) == "runs" else "runs", time.strftime("%Y%m%d-%H%M%S.toml"))
    typed = f"bs run {path}" + "".join(f" --only {o}" for o in only) + (f" --from {from_}" if from_ else "")
    notes = [f"{typed}, {time.strftime('%Y-%m-%d %H:%M:%S')}, bootstrapper {version('bootstrapper')}, on {platform.node()}"]
    ran = []
    for n, f, s in todo:
        cli_echo(f"Running {s.label} from {f}", s.kind)
        name, start = f"{os.path.basename(f)} [{s.label}]", time.time()
        try:
            if s.kind == "refine":
                run_refine(ctx, s)
            else:
                command = {"train": train, "predict": predict, "segment": segment, "evaluate": evaluate}[s.kind]
                ctx.invoke(command, config_file=f, step=None if s.name == s.kind else s.name)
        except Exception as e:
            error = e.message if isinstance(e, click.ClickException) else f"{type(e).__name__}: {e}"
            write_receipt(receipt, [*notes, f"{name}: failed after {_took(start)}: {error}"], volumes, [*ran, s])
            raise
        notes.append(f"{name}: done, {_took(start)}")
        ran.append(s)
        write_receipt(receipt, notes, volumes, ran)
        click.echo(f"View: bs view {path} {n}")


def refuse_stale(path):
    """Refuse a round whose run tails differ from the ones bs prepare derives now."""
    root = prepared_round(path)
    changed = stale(root) if root else []
    if not changed:
        return
    lines = [f"  {os.path.basename(f)} [{table}] {k}: {_was(a, b)}"
             for f, _, _, keys in changed for table, ks in keys.items() for k, (a, b) in ks.items()]
    more = [f"  and {len(lines) - 3} more"] if len(lines) > 3 else []
    n = len(changed)
    raise click.ClickException("\n".join([
        f"{root}: {n} run tail{'s are' if n > 1 else ' is'} out of date with the keys above {'them' if n > 1 else 'it'}. Nothing ran.",
        *lines[:3], *more, f"Run: bs prepare {root}"]))


def _was(old, new):
    """'old is now new', without the parts between / and -- that both share at either end."""
    text = ("none" if v is None else ", ".join(map(str, v)) if isinstance(v, list) else str(v) for v in (old, new))
    a, b = (re.split(r"(?=/|--)", t) for t in text)
    i = len(os.path.commonprefix([a, b]))
    j = len(os.path.commonprefix([a[i:][::-1], b[i:][::-1]]))
    return " is now ".join("..." * (i > 0) + "".join(p[i:len(p) - j]) + "..." * (j > 0) for p in (a, b))


def plan(path, todo, volumes):
    """The dry-run lines: what each step reads and writes. Refuses a read that is not there and
    that no earlier step writes."""
    containers = {v.keys["raw"].rsplit(".zarr", 1)[0] + ".zarr/" for v in volumes.values()}
    base = containers.pop() if len(containers) == 1 else ""
    lines = [f"{path}: {_count(len(todo), 'step')}" + (f" on {_count(len(volumes), 'volume')}" if volumes else "")
             + "." + (f" Paths are in {base[:-1]} unless absolute." if base else "")]
    written = []  # (path prefix, step number) of what earlier steps write
    for n, f, s in todo:
        for vol, config in runs(s):
            reads = []
            for r in (p for key in READS[s.kind] for p in paths(config, key)):
                src = max(((len(w), m) for w, m in written if (r + "/").startswith(w)), default=None)
                if src:
                    reads.append(f"{r.removeprefix(base)} from {src[1]}")
                elif r.startswith(MODEL_DIR):
                    reads.append(f"{os.path.basename(r)} (shipped)")
                elif os.path.exists(r) or os.path.exists(r + ".ckpt"):  # predict adds .ckpt to a checkpoint
                    reads.append(r.removeprefix(base))
                else:
                    raise click.ClickException(f"{os.path.basename(f)} [{s.label}]{' ' + vol if vol else ''} reads {r}: "
                                               "nothing there, and no earlier step writes it. Nothing ran.")
            writes = [p + "_*" if s.kind == "refine" else p for key in WRITES[s.kind] for p in paths(config, key)]
            written += [(w[:-1] if w.endswith("*") else w + "/", n) for w in writes]
            ops = f": {', '.join(s.keys['ops'])}" if s.kind == "refine" else ""
            lines += [f"{n} {s.label}{' on ' + vol if vol else ''}{ops}",
                      *([f"   reads  {', '.join(reads)}"] if reads else []),
                      *([f"   writes {', '.join(w.removeprefix(base) for w in writes)}"] if writes else [])]
    return lines + ["Every input exists or comes from an earlier step. Nothing ran."]


def write_receipt(path, notes, volumes, steps):
    """A round file of the steps that ran, each with the keys it read, below one note per step."""
    doc = {"volumes": {n: v.keys for n, v in volumes.items()}} if volumes else {}
    for s in steps:
        table = ops_beside(s.keys) | {v: ops_beside(keys) for v, keys in s.volumes.items()}
        doc.setdefault(s.kind, {}).update(table if s.name == s.kind else {s.name: table})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(file_text("\n# ".join([*notes, "The keys below are the ones each step read. bs run on this file repeats the steps."]), doc))


def _took(start):
    seconds = time.time() - start
    return f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _count(n, word):
    return f"{n} {word}{'s' * (n != 1)}"


def run_refine(ctx, step):
    """Run a refine step's ops in order on each volume; each op's output feeds the next. A
    volume's op table, such as [run.refine.<volume>.mask], merges over the step's."""
    ops = step.keys.get("ops", {})
    if not ops:
        raise click.ClickException(f"[{step.label}] has no ops; ops are {', '.join(OPS)}")
    configs = runs(step)
    for _, config in configs:
        for op in ops:
            keys = config["ops"][op]
            command = refine.commands["morph" if op in MORPH_OPS else op]
            names = {p.name for p in command.params} - {"in_array", "op"}
            if "dry_run" in keys and "dry_run" in names:
                raise click.ClickException(f"[{step.label}.{op}] has no key 'dry_run'; a chain has no dry run, so run bs refine {op} --dry-run")
            refuse_unknown(f"[{step.label}.{op}]", keys, names - {"dry_run"})
    for _, config in configs:
        array = config.get("in_array")
        if array is None:
            raise click.ClickException(f"[{step.label}] needs in_array")
        defaults = {"num_workers": config["num_workers"]} if "num_workers" in config else {}
        for op in ops:
            keys = defaults | config["ops"][op]
            if op in MORPH_OPS:
                keys["op"] = op
                op = "morph"
            array = ctx.invoke(refine.commands[op], in_array=array, **keys)


