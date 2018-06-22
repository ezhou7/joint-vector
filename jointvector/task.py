import json
import numpy as np

from jointvector.path import get_props_path


class NLPTask:
    def __init__(self, props_file_name):
        self.set_props(props_file_name)
        self.label_to_idx = {label: i for i, label in enumerate(getattr(self, "output_labels"))}

    def set_props(self, props_file_name):
        with open(get_props_path(props_file_name), "r") as fin:
            props = json.load(fin)

        for key, value in props:
            setattr(self, key, value)

    def get_num_labels(self):
        return len(getattr(self, "output_labels"))

    def get_label(self, index):
        return getattr(self, "output_labels")[index]

    def generate_labels(self, sentences):
        return np.array([
            getattr(token, getattr(self, "task_tag"))
            for sentence in sentences
            for token in sentence.tokens
        ])
