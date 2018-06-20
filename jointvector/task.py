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
    def auxiliary_arch(self):
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

    def auxiliary_arch(self):
        pass
