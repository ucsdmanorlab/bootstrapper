import argparse
import subprocess
import os

this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def run_subprocess(this_dir, script, *args):
    subprocess.run(["python", os.path.join(this_dir, script), *args], check=True)

def main():
    print("Bootstrapper CLI Test")
    run_subprocess(this_dir, "prepare_configs.py")

if __name__ == "__main__":
    main()