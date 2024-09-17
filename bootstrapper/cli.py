import argparse
import subprocess
import os

def run_subprocess(script, *args):
    subprocess.run(["python", os.path.join(this_dir, script), *args], check=True)

def main():
    print("Bootstrapper CLI Test")
    subprocess.run(["python", os.path.join(this_dir, "prepare_configs.py")], check=True)

if __name__ == "__main__":
    this_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    main()