import click

def create_style_palette(base_color):
    return {
        "info": {"fg": base_color, "dim": True},
        "prompt": {"fg": base_color, "italic": True},
        "success": {"fg": f"bright_{base_color}", "bg": "green", "bold": True, "dim": True},
        "error": {"fg": f"bright_{base_color}", "bg": "red", "bold": True},
        "warning": {"fg": f"bright_{base_color}", "bg": "yellow", "bold": True},
    }

STYLES = {
    "default": create_style_palette("white"),
    "prepare": create_style_palette("cyan"),
    "train": create_style_palette("green"),
    "predict": create_style_palette("yellow"),
    "segment": create_style_palette("red"),
    "evaluate": create_style_palette("magenta"),
    "filter": create_style_palette("blue"),
}

def get_style(style, stype='info'):
    return STYLES.get(style, STYLES['default']).get(stype, {})

def cli_echo(text, style='default', stype='info'):
    click.secho(text, **get_style(style, stype))

def cli_prompt(text, style='default', stype='prompt', **kwargs):
    style = get_style(style, stype)
    return click.prompt(click.style(text, **style), prompt_suffix=click.style(" >>> ", **style), **kwargs)

def cli_confirm(text, style='default', stype='prompt', **kwargs):
    style = get_style(style, stype)
    return click.confirm(click.style(text, **style), prompt_suffix=click.style(" >>> ", **style), **kwargs)