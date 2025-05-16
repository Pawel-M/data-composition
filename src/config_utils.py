import os
import sys
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return str(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def get(self, item, default=None):
        return self.__dict__.get(item, default)


def load_config(config_path, return_struct=True):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if return_struct:
        return Struct(**config)

    return config


def load_configs(configs_paths, return_struct=True):
    config_dict = {}
    for config_path in configs_paths:
        current_config = load_config(config_path, return_struct=False)
        config_dict.update(current_config)

    if return_struct:
        return Struct(**config_dict)

    return config_dict


def expand_path(path):
    # home_path = os.environ.get(home_variable_name)
    # if home_path is not None:
    #     path = os.path.join(home_path, path)

    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return path
