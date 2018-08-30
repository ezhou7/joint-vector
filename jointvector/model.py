import json
import logging
import numpy as np

from jointvector.builder import EmbeddingSystemBuilder
from jointvector.data import generate_features
from jointvector.path import get_task_props_path
from jointvector.timer import stopwatch


class EmbeddingSystem:
    def __init__(self, tasks, word2vec):
        self.tasks = tasks
        self.word2vec = word2vec

        with open(get_task_props_path("embedding-props.json"), "r") as fin:
            embedding_sys_props = json.load(fin)

        self.words_window_size = embedding_sys_props["words_window_size"]
        self.ngram_1_filters = embedding_sys_props["ngram_1_filters"]
        self.ngram_2_filters = embedding_sys_props["ngram_2_filters"]
        self.conv_3_filters = embedding_sys_props["conv_3_filters"]

        model_builder = EmbeddingSystemBuilder(
            word2vec.get_dimension(),
            self.words_window_size,
            self.ngram_1_filters,
            self.ngram_2_filters,
            self.conv_3_filters,
            tasks
        )

        self.models = model_builder.build()

    def train(self, trn_data, dev_data, nb_epochs=5, batch_size=32):
        xtrn = generate_features(trn_data, self.word2vec, self.words_window_size)
        xdev = generate_features(dev_data, self.word2vec, self.words_window_size)

        best_epoch = 1
        best_trn_scores = [0.0] * len(self.tasks)
        best_dev_scores = [0.0] * len(self.tasks)

        stopwatch.start("epoch_training")

        for epoch in range(1, nb_epochs):
            print("Epoch {}".format(epoch))

            curr_trn_scores = [0.0] * len(self.tasks)
            curr_dev_scores = [0.0] * len(self.tasks)

            for i, (task, model) in enumerate(zip(self.tasks, self.models)):
                ytrn = task.generate_labels(trn_data)
                ydev = task.generate_labels(dev_data)

                stopwatch.start("{}_training".format(task.task_name))
                h = model.fit(x=xtrn, y=ytrn, validation_data=(xdev, ydev), batch_size=batch_size, epochs=1, verbose=0)
                training_time = stopwatch.end("{}_training".format(task.task_name))

                curr_trn_scores[i] = h.history["sparse_categorical_accuracy"]
                curr_dev_scores[i] = h.history["val_sparse_categorical_accuracy"]

                print(
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

        print(
            "Summary - total time: {0:.2f}, Best Scores @ Epoch {1:d}".format(
                stopwatch.end("epoch_training"),
                best_epoch
            )
        )

        for task, best_trn_score, best_dev_score in zip(self.tasks, best_trn_scores, best_dev_scores):
            print("{0}: Trn - {1:.2f}, Dev - {2:.2f}".format(task.task_name, best_trn_score, best_dev_score))

    def predict(self, X):
        return {task.task_name: model.predict(X) for task, model in zip(self.tasks, self.models)}
