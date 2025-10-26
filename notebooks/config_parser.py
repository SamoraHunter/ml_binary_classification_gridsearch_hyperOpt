import yaml
from pathlib import Path

def load_config(config_path: str = 'config.yml'):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.
                           Defaults to 'config.yml' in the project root.

    Returns:
        dict: A dictionary containing the configuration.
    """
    # Assume config_path is relative to the project root
    full_path = Path(__file__).resolve().parents[1] / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")

    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)
    return config