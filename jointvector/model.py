import json
from jointvector.builder import EmbeddingSystemBuilder

from jointvector.path import get_props_path


class EmbeddingSystem:
    def __init__(self, tasks):
        self.tasks = tasks

        # TODO: implement auto reading of embedding dimensions
        self.embedding_dims = 0

        with open(get_props_path("embedding-props.json"), "r") as fin:
            embedding_sys_props = json.load(fin)

        model_builder = EmbeddingSystemBuilder(
            self.embedding_dims,
            embedding_sys_props["words_window_size"],
            embedding_sys_props["ngram_1_filters"],
            embedding_sys_props["ngram_2_filters"],
            embedding_sys_props["conv_3_filters"],
            tasks
        )

        self.model = model_builder.build()

    def train(self):
        pass
