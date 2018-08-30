from jointvector.visual.builder import VisualBranchBuilder
from jointvector.path import get_task_props_path


class VisualBranch:
    def __init__(self):
        dense_net_props = get_task_props_path("dense-net-169-props.json")

        self.block_sizes = dense_net_props["block_sizes"]
        self.growth_rate = dense_net_props["growth_rate"]
        self.compression_rate = dense_net_props["compression_rate"]
        self.use_bottlenecks = dense_net_props["use_bottlenecks"]

        model_builder = VisualBranchBuilder(
            self.block_sizes,
            # TODO: fill in image dimensions
            0,
            self.growth_rate,
            self.compression_rate,
            self.use_bottlenecks
        )
        self.model = model_builder.build()

    def train(self, trn_data, dev_data, nb_epochs=5, batch_size=32):
        # TODO: synchronize image training data with text training data
        for epoch in range(nb_epochs):
            pass

    def predict(self, tst_data):
        pass
