# fedml_api/standalone/models/__init__.py

import os
import importlib


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('./src/models')
            if not model.find('__') > -1 and 'py' in model]


names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


def get_model(nets_list, args, transform):
    return names[args.model](nets_list, args, transform)
