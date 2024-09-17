import argparse
import yaml
import subprocess
import os

def extract_setup_dir(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['setup_dir']

def run_training(yaml_file):
    setup_dir = extract_setup_dir(yaml_file)
    train_script = os.path.join(setup_dir, 'train.py')
    
    # Run the training script with the YAML file as an argument
    command = ["python", train_script, yaml_file]
    result = subprocess.run(command, check=True)
    
    if result.returncode == 0:
        print("Training completed successfully.")
    else:
        print("Training failed.")

def main():
    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument('yaml_file', type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    run_training(args.yaml_file)

if __name__ == '__main__':
    main()