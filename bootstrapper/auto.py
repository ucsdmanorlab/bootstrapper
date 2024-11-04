import click
import os
import glob

def get_round_configs(base_dir):
    configs = []

    round_dirs = sorted(glob.glob(os.path.join(base_dir, "round_*")))

    for round_dir in round_dirs:
        configs.append(
            {
                "train": glob.glob(os.path.join(round_dir, "run", "train*"))[0],
                "predict": glob.glob(os.path.join(round_dir, "run", "pred*")),
                "segment": glob.glob(os.path.join(round_dir, "run", "seg*")),
                "eval": glob.glob(os.path.join(round_dir, "run", "eval*")),
                "filter": glob.glob(os.path.join(round_dir, "run", "filter*")),
            }
        )

    return configs


def run_auto(base_dir):
    from bootstrapper.train import run_training
    from bootstrapper.predict import run_prediction
    from bootstrapper.segment import run_segmentation
    from bootstrapper.evaluate import run_evaluation
    from bootstrapper.filter import run_filter

    rounds = get_round_configs(base_dir)

    for r in rounds:
        run_training(r['train'])

        for v in r['predict']:
            run_prediction(v)

        for v in r['segment']:
            run_segmentation(v)

        for v in r['eval']:
            run_evaluation(v)

        for v in r['filter']:
            run_filter(v)


@click.command()
def auto():
    """Run auto bootstrapper in the current directory."""
    run_auto(os.getcwd())
