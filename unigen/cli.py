import sys
from enum import Enum, unique
import yaml

USAGE = ("-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "| unigen-cli generation <config.yaml> -h: start generation           |\n"
    + "| unigen-cli analysis <config.yaml> -h: start analysis               |\n"
    + "| unigen-cli evaluation <config.yaml> -h: start evaluation           |\n"
    + "| unigen-cli augmentation <config.yaml> -h: start augmentation       |\n"
    + "-" * 70
)

@unique
class Command(str, Enum):
    GEN = 'generation'
    ANA = 'analysis'
    EVAL = 'evaluation'
    AUG = 'augmentation'
    HELP = 'help'

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generation(config):
    print("Starting generation process with config:", config)


def analysis(config):
    print("Starting analysis process with config:", config)

def evaluation(config):
    print("Starting evaluation process with config:", config)
    from utils import generation
    Generator = generation.LLMGeneration(config)
    Generator.generation_results()

def augmentation(config):
    print("Starting augmentation process with config:", config)

def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
        
    command = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    if command in {Command.GEN, Command.ANA, Command.EVAL, Command.AUG} and config_path:
        config = load_config(config_path)
        if command == Command.GEN:
            generation(config)
        elif command == Command.ANA:
            analysis(config)
        elif command == Command.EVAL:
            evaluation(config)
        elif command == Command.AUG:
            augmentation(config)
    elif command == Command.HELP:
        print(USAGE)
    else:
        print(USAGE)
        print(f"Invalid command or missing config path: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()