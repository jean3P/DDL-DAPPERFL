# fedml_api/standalone/datasets/__init__.py

from pathlib import Path
import importlib
import inspect
import os

from argparse import Namespace
from .utils.federated_dataset import FederatedDataset
from .utils.public_dataset import PublicDataset

# 1) Points to the "src/datasets" directory (the same directory this file is in).
DATASETS_DIR = Path(__file__).resolve().parent

# 2) Base import path for all your dataset modules
MODULE_BASE = "datasets"


def get_all_models():
    """
    Returns a list of Python module *names* in src/datasets,
    excluding __init__.py and other non-.py files.
    """
    model_names = []
    for item in DATASETS_DIR.iterdir():
        # Check that it's a file, ends with .py, not __init__.py
        if (
                item.is_file()
                and item.suffix == ".py"
                and not item.name.startswith("__")
        ):
            # item.stem is the filename without .py (e.g., "my_dataset_1")
            model_names.append(item.stem)
    return model_names


# Dictionaries to store your dataset classes by name
Priv_NAMES = {}
Pub_NAMES = {}

# Go through each module in src/datasets
for model_name in get_all_models():
    # Dynamically import, e.g. "src.datasets.my_dataset_1"
    mod = importlib.import_module(f"{MODULE_BASE}.{model_name}")

    # Find classes that inherit from FederatedDataset
    federated_classes = [
        cls_name
        for cls_name in dir(mod)
        if (
                "type" in str(type(getattr(mod, cls_name))) and
                "FederatedDataset" in str(inspect.getmro(getattr(mod, cls_name))[1:])
        )
    ]
    for cls_name in federated_classes:
        c = getattr(mod, cls_name)
        Priv_NAMES[c.NAME] = c  # c.NAME is your custom property

    # Find classes that inherit from PublicDataset
    public_classes = [
        cls_name
        for cls_name in dir(mod)
        if (
                "type" in str(type(getattr(mod, cls_name))) and
                "PublicDataset" in str(inspect.getmro(getattr(mod, cls_name))[1:])
        )
    ]
    for cls_name in public_classes:
        c = getattr(mod, cls_name)
        Pub_NAMES[c.NAME] = c


def get_prive_dataset(args: Namespace) -> FederatedDataset:
    assert args.dataset in Priv_NAMES, f"{args.dataset} not found in Priv_NAMES"
    return Priv_NAMES[args.dataset](args)


def get_public_dataset(args: Namespace) -> PublicDataset:
    assert args.public_dataset in Pub_NAMES, f"{args.public_dataset} not found in Pub_NAMES"
    return Pub_NAMES[args.public_dataset](args)
