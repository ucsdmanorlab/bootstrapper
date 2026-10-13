"""The TOML text of a config file. A key of params.KEYS carries what it does, and its off
keys and tables show commented out."""
import json
import re

from .params import KEYS

COLUMN, WIDTH = 35, 100  # an inline comment starts at COLUMN; one that runs past WIDTH goes above


def file_text(title, doc):
    out = [f"# {title}"]
    _table([], doc, out)
    return "\n".join(out) + "\n"


def _table(path, doc, out):
    """doc None: the table is off, so its header and keys show commented out."""
    off, doc, name = doc is None, doc or {}, ".".join(path)
    wild = ".".join([path[0], "*", *path[2:]]) if len(path) > 1 else None
    spec = KEYS.get(name) or KEYS.get(wild, {})
    keys = {k: v for k, v in doc.items() if not isinstance(v, dict) and not _rows(v)}
    if keys or spec or not doc:
        out += ["", f"{'# ' * off}[{_dotted(path)}]"]
    for k, v in keys.items():
        out += _line(k, v, spec[k].help if k in spec else "")
    for k, key in spec.items():
        if k not in keys and (key.off or off):
            out += _line(k, key.value, key.help, off=True)
    for k, v in doc.items():
        for row in _rows(v):
            out += ["", f"[[{_dotted(path + [k])}]]", *(line for rk, rv in row.items() for line in _line(rk, rv))]
    # the tables of KEYS come first, and one that doc lacks shows off
    tables = {child: None for parent, _, child in (n.rpartition(".") for n in KEYS) if parent and parent == name and child != "*"}
    for k, v in (tables | {k: v for k, v in doc.items() if isinstance(v, dict)}).items():
        _table(path + [k], v, out)


def _line(key, value, help="", off=False):
    text = f"{'# ' * off}{_bare(key)} = {_value(value)}"
    if not help:
        return [text]
    if len(text) < COLUMN and COLUMN + 2 + len(help) <= WIDTH:
        return [f"{text:<{COLUMN}}# {help}"]
    return [f"# {help}", text]


def _rows(value):
    """An array of tables, as [[train.<model>.samples]]."""
    return value if isinstance(value, list) and value and all(isinstance(v, dict) for v in value) else []


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return f"[{', '.join(map(_value, v))}]"
    return repr(v)


def _bare(key):
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key, ensure_ascii=False)


def _dotted(path):
    return ".".join(map(_bare, path))
