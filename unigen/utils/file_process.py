import json
import os

import yaml


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_yaml(data, file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def check_and_rename_file(path):
    """
    Check if a file exists at the given path. If it does, rename it by adding
    a number (n) to its name before the file extension.

    Args:
    - path (str): The original file path.

    Returns:
    - str: The new file path, with an appended number if the file existed.
    """
    if os.path.exists(path):
        # Split the path into directory, base, and extension
        directory, filename = os.path.split(path)
        base, extension = os.path.splitext(filename)

        # Initialize counter
        counter = 1

        # Generate new file name with a counter until it's unique
        new_filename = f"{base}({counter}){extension}"
        new_path = os.path.join(directory, new_filename)
        while os.path.exists(new_path):
            counter += 1
            new_filename = f"{base}({counter}){extension}"
            new_path = os.path.join(directory, new_filename)

        return new_path
    else:
        # If the file doesn't exist, return the original path
        return path

