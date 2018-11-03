import abc
from keras.layers import Input


class NeuralArchitectureBuilder:
    @abc.abstractmethod
    def build(self):
        return


class InputBuilder(NeuralArchitectureBuilder):
    def __init__(self, num_rows, num_cols):
        super().__init__()
        self.shape = (num_rows, num_cols)

    def build(self):
        return Input(shape=self.shape)
