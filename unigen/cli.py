import os
import random
import subprocess
import sys
from enum import Enum, unique


USAGE = ("-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "| unigen-cli generation -h: start generation                         |\n"
    + "| unigen-cli analysis -h: start analysis                             |\n"
    + "| unigen-cli evaluation -h: start evaluation                         |\n"
    + "| unigen-cli augmentation -h: start augmentation                     |\n"
    + "-" * 70
)

@unique
class Command(str, Enum):
    GEN = 'generation'
    ANA = 'analysis'
    EVAL = 'evaluation'
    AUG = 'augmentation'
    HELP = 'help'


def main():
    command = sys.argv.pop(1)
    if command == Command.GEN:
        generation()
    elif command == Command.ANA:
        analysis()
    elif command == Command.EVAL:
        evaluation()
    elif command == Command.AUG:
        augmentation()
    elif command == Command.HELP:
        print(USAGE)
    else:
        print(USAGE)
        print(f"Invalid command: {command}")
        sys.exit(1)
