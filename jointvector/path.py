import os

from jointvector import ROOT_DIR


def get_resources_path():
    return os.path.join(ROOT_DIR, "resources/")


def get_props_path():
    return os.path.join(ROOT_DIR, "props/")


def get_task_props_path(props_filename):
    return os.path.join(get_props_path(), props_filename)
