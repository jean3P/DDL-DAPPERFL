# src/utils/conf.py

import random
import torch
import numpy as np
from pathlib import Path


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def get_project_root() -> Path:
    """
    Returns the absolute path of the project root by going up from this file.
    Adjust the number of `.parent` as needed.
    """
    return Path(__file__).resolve().parent.parent.parent


def data_path() -> str:
    return str(get_project_root() / 'resources')


def base_path() -> str:
    return './data/'


def checkpoint_path() -> str:
    return str(get_project_root() / 'checkpoint')


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
