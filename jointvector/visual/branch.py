from jointvector.visual.builder import VisualBranchBuilder


class VisualBranch:
    def __init__(self):
        # TODO: fill in properties in VisualBranchBuilder
        model_builder = VisualBranchBuilder()
        self.model = model_builder.build()

    def train(self, trn_data, dev_data, nb_epochs=5, batch_size=32):
        pass

    def predict(self, tst_data):
        pass
