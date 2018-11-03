import json
import numpy as np

from keras.layers import Input

from jointvector.path import get_task_props_file_path
from jointvector.util import read_json_file
from jointvector.visual.branch import VisualBranch


class NLPTask:
    def __init__(self, props_file_name):
        props = read_json_file(get_task_props_file_path(props_file_name))

        self.task_name = props["task_name"]
        self.task_tag = props["task_tag"]
        self.output_labels = props["output_labels"]
        self.activation = props["activation"]
        self.optimizer = props["optimizer"]
        self.loss_func = props["loss_func"]
        self.metrics = props["metrics"]

        self.label_to_idx = {label: i for i, label in enumerate(getattr(self, "output_labels"))}

    def get_num_labels(self):
        return len(self.output_labels)

    def get_label(self, index):
        return self.output_labels[index]

    def get_label_index(self, label):
        return self.label_to_idx[label]

    def generate_label_indices_for_instances(self, sentences):
        return np.array([
            self.get_label_index(getattr(token, self.task_tag))
            for sentence in sentences
            for token in sentence.tokens
        ])

    def get_task_input_neural_arch(self, shape):
        return Input(shape=shape)


# TODO: implement this class
class CVTask:
    def __init__(self, props_file_name):
        self.set_props(props_file_name)
        self.visual_branch = VisualBranch()

    def set_props(self, props_file_name):
        with open(get_task_props_file_path(props_file_name), "r") as fin:
            props = json.load(fin)

        for key, value in props.items():
            setattr(self, key, value)

    def get_true_outputs_for_instances(self, data):
        # TODO: implement this method
        pass

    def get_task_input_neural_arch(self):
        return self.visual_branch.get_penultimate_layer()
