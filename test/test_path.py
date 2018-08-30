import os

from jointvector.path import get_props_dir_path


def test_resources_path():
    res_path = get_props_dir_path()
    assert os.path.exists(res_path), "Resource path does not exist: {}".format(res_path)
