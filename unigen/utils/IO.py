import warnings
warnings.filterwarnings("ignore")

original_print = print
original_input = input

# Color code definitions
GREEN = '\033[92m'
RED = '\033[31m'
BLUE = '\033[34m'
END_COLOR = '\033[0m'  # Unified end color code for resetting color

# Color code dictionary for easy look-up
COLOR_CODES = {
    'GREEN': GREEN,
    'RED': RED,
    'BLUE': BLUE
}


def input(text, color=None):
    """Custom input function with optional color."""
    color_code = COLOR_CODES.get(color.upper(), "") if color else ""
    return original_input(f"{color_code}{text}{END_COLOR}")


def print(text, color=None):
    """Custom print function with optional color."""
    color_code = COLOR_CODES.get(color.upper(), END_COLOR) if color else END_COLOR
    original_print(f"{color_code}{text}{END_COLOR}")
