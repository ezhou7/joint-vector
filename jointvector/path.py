import os

from setup import ROOT_DIR


def get_resources_path():
    return os.path.join(ROOT_DIR, "resources/")


def get_props_path(props_filename):
    return os.path.join(get_resources_path(), props_filename)
