import json
from abc import ABC, abstractmethod
from collections import namedtuple

from jointvector.path import get_props_path


def json_to_task(task_props_name):
    return namedtuple(task_props_name, "task_name output_labels activation optimizer loss_func metrics")


class NLPTask(ABC):
    def __init__(self, task_name, output_labels, activation, optimizer, loss_func, metrics):
        self.task_name = task_name
        self.output_labels = output_labels
        self.label_to_idx = {label: i for i, label in enumerate(self.output_labels)}

        self.activation = activation
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = metrics

    def get_num_labels(self):
        return len(self.output_labels)

    def get_label(self, index):
        return self.output_labels[index]

    @abstractmethod
    def generate_data(self, sentences):
        raise NotImplementedError()


class POSTask(NLPTask):
    def __init__(self):
        with open(get_props_path("pos-task-props.json"), "r") as fin:
            pos_props = json.load(fin, object_hook=lambda d: json_to_task("POSTaskProps")(*d.values()))

        super().__init__(
            pos_props.task_name,
            pos_props.output_labels,
            pos_props.activation,
            pos_props.optimizer,
            pos_props.loss_func,
            pos_props.metrics
        )

    def generate_data(self, sentences):
        pass


class DEPTask(NLPTask):
    def __init__(self):
        with open(get_props_path("dep-task-props.json"), "r") as fin:
            dep_props = json.load(fin, object_hook=lambda d: json_to_task("DEPTaskProps")(*d.values()))

        super().__init__(
            dep_props.task_name,
            dep_props.output_labels,
            dep_props.activation,
            dep_props.optimizer,
            dep_props.loss_func,
            dep_props.metrics
        )

    def generate_data(self, sentences):
        pass


class NERTask(NLPTask):
    def __init__(self):
        with open(get_props_path("ner-task-props.json"), "r") as fin:
            ner_props = json.load(fin, object_hook=lambda d: json_to_task("NERTaskProps")(*d.values()))

        super().__init__(
            ner_props.task_name,
            ner_props.output_labels,
            ner_props.activation,
            ner_props.optimizer,
            ner_props.loss_func,
            ner_props.metrics
        )

    def generate_data(self, sentences):
        pass
