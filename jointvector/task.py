import json
from abc import ABC, abstractmethod


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
    def __init__(self, task_name, output_labels, activation, optimizer, loss_func, metrics):
        json.load()
        super().__init__(task_name, output_labels, activation, optimizer, loss_func, metrics)

    def auxiliary_arch(self):
        pass
