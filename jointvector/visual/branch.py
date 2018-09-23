from jointvector.visual.builder import VisualBranchBuilder
from jointvector.path import get_task_props_file_path
from jointvector.util import read_json_file


class VisualBranch:
    def __init__(self):
        dense_net_props_path = get_task_props_file_path("dense-net-169-props.json")
        dense_net_props = read_json_file(dense_net_props_path)

        self.block_sizes = dense_net_props["block_sizes"]
        self.growth_rate = dense_net_props["growth_rate"]
        self.compression_rate = dense_net_props["compression_rate"]
        self.use_bottlenecks = dense_net_props["use_bottlenecks"]

        model_builder = VisualBranchBuilder(
            self.block_sizes,
            self.growth_rate,
            self.compression_rate,
            self.use_bottlenecks
        )
        self.model = model_builder.build()

    def train(self, trn_data, dev_data, nb_epochs=5, batch_size=32):
        for epoch in range(nb_epochs):
            # TODO: find out how to predict bounding boxes/annotations in an image
            self.model.fit(
                trn_data["images"],
                trn_data["image_annotations"],
                validation_data=(dev_data["images"], dev_data["image_annotations"]),
                batch_size=batch_size,
                epochs=1,
                verbose=0
            )

    def predict(self, tst_data):
        pass
