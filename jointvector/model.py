import json
import logging
import numpy as np
from jointvector.builder import EmbeddingSystemBuilder

from jointvector.path import get_props_path
from jointvector.timer import stopwatch


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

        self.models = model_builder.build()

    def train(self, xtrn, ytrn, xdev, ydev, nb_epochs=5, batch_size=32):
        best_epoch = 1
        best_trn_scores = [0.0] * len(self.tasks)
        best_dev_scores = [0.0] * len(self.tasks)

        for epoch in range(nb_epochs, 1):
            logging.info("Epoch {}".format(epoch))
            stopwatch.start("epoch_training")

            curr_trn_scores = [0.0] * len(self.tasks)
            curr_dev_scores = [0.0] * len(self.tasks)

            for i, (task, model) in enumerate(zip(self.tasks, self.models)):
                stopwatch.start("{}_training".format(task.task_name))
                h = model.fit(x=xtrn, y=ytrn, validation_data=(xdev, ydev), batch_size=batch_size, epochs=1, verbose=0)
                training_time = stopwatch.end("{}_training".format(task.task_name))

                curr_trn_scores[i] = h.history["sparse_categorical_accuracy"]
                curr_dev_scores[i] = h.history["val_sparse_categorical_accuracy"]

                logging.info(
                    "{} - trn time: {.2f}, Trn - ls {.2f} ac {.2f}, Dev - ls {.2f} ac {.2f}".format(
                        task.task_name,
                        training_time,
                        h.history["loss"],
                        h.history["sparse_categorical_accuracy"],
                        h.history["val_loss"],
                        h.history["val_sparse_categorical_accuracy"]
                    )
                )

            if not np.any(curr_dev_scores <= best_dev_scores):
                best_epoch = epoch
                best_trn_scores = curr_trn_scores
                best_dev_scores = curr_dev_scores

        logging.info(
            "Summary - total time: {.2f}, Best Scores @ Epoch {1:d}".format(
                stopwatch.end("epoch_training"),
                best_epoch
            )
        )

        for task, best_trn_score, best_dev_score in zip(self.tasks, best_trn_scores, best_dev_scores):
            logging.info("{}: Trn - {.2f}, Dev - {.2f}".format(task.task_name, best_trn_score, best_dev_score))

    def predict(self, X):
        return {task.task_name: model.predict(X) for task, model in zip(self.tasks, self.models)}
