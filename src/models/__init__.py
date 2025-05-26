import os
import importlib


def get_all_models():
    # Get the absolute path to the models directory
    current_file = os.path.abspath(__file__)
    models_dir = os.path.dirname(current_file)

    model_names = []
    for item in os.listdir(models_dir):
        # Skip hidden files and directories as well as non-model directories
        if item.startswith('__'):
            continue
        # Exclude "utils" directory (or any other non-model folders)
        if item.lower() in ['utils']:
            continue
        full_path = os.path.join(models_dir, item)
        # If item is a .py file (e.g., dapperfl.py)
        if os.path.isfile(full_path) and item.endswith('.py'):
            model_names.append(item.split('.')[0])
        # If item is a directory with an __init__.py file (e.g., fedsr/)
        elif os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, '__init__.py')):
            model_names.append(item)
    return model_names


names = {}
for model in get_all_models():
    # Get the absolute path to the models directory
    current_file = os.path.abspath(__file__)
    models_dir = os.path.dirname(current_file)
    full_path = os.path.join(models_dir, model)

    # If the item is a directory, check if it contains a "model.py" file.
    if os.path.isdir(full_path):
        model_file = os.path.join(full_path, 'model.py')
        if os.path.exists(model_file):
            mod = importlib.import_module('models.' + model + '.model')
        else:
            mod = importlib.import_module('models.' + model)
    else:
        mod = importlib.import_module('models.' + model)

    # Look for a class in the module whose name (lowercased and with underscores removed) equals the model name.
    class_name = None
    for attr, value in mod.__dict__.items():
        if isinstance(value, type) and attr.lower().replace('_', '') == model.lower().replace('_', ''):
            class_name = attr
            break

    if class_name is None:
        raise ValueError(f"Could not find a matching class for model {model} in module {mod}")

    names[model] = getattr(mod, class_name)


def get_model(nets_list, args, transform):
    return names[args.model](nets_list, args, transform)
