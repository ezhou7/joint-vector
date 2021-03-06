from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Reshape, concatenate


def conv_pool_pair(num_filters, kernel_size, conv_activation, pool_size):
    conv_layer = Conv2D(filters=num_filters, kernel_size=kernel_size, activation=conv_activation)
    pool_layer = MaxPool2D(pool_size=pool_size)

    return conv_layer, pool_layer


class EmbeddingSystemBuilder:
    def __init__(self, embedding_dims, words_window_size, ngram_1_filters, ngram_2_filters, conv_3_filters, tasks):
        self.embedding_dims = embedding_dims
        self.words_window_size = words_window_size
        self.ngram_1_filters = ngram_1_filters
        self.ngram_2_filters = ngram_2_filters
        self.conv_3_filters = conv_3_filters
        self.tasks = tasks

    def build(self):
        task_inputs = [task.get_task_input_neural_arch() for task in self.tasks]
        model_input, output_embedding_layer = self.__build_arch()
        task_outputs = [
            Dense(
                units=task.get_num_labels(),
                activation=task.activation
            )(output_embedding_layer)
            for task in self.tasks
        ]

        models = [Model(inputs=task_inputs, outputs=task_output) for task_output in task_outputs]

        for task, model in zip(self.tasks, models):
            model.compile(optimizer=task.optimizer, loss=task.loss_func, metrics=task.metrics)

        return models

    def __build_arch(self):
        num_ngrams_1 = len(self.ngram_1_filters)
        num_ngrams_2 = len(self.ngram_2_filters)

        if self.words_window_size < num_ngrams_1:
            raise Exception("Too many first level n-gram filters compared to window of context words:\n" +
                            "Window size: {}\n".format(self.words_window_size) +
                            "Level 1 n-gram filters: {}\n".format(num_ngrams_1))

        if num_ngrams_1 < num_ngrams_2:
            raise Exception("Too many second level n-gram filters compared to first level n-gram filters:\n" +
                            "Level 1 n-gram filters: {}\n".format(num_ngrams_1) +
                            "Level 2 n-gram filters: {}\n".format(num_ngrams_2))

        # if num_ngrams_2 < self.conv_3_filters:
        #     raise Exception("Too many third level conv filters compared to second level n-gram filters:\n" +
        #                     "Level 2 n-gram filters: {}\n".format(num_ngrams_2) +
        #                     "Level 3 conv filters: {}\n".format(self.conv_3_filters))

        # ----------- create layers ----------- #

        # TODO: remove model_input b/c unnecessary since tasks provide their own neural architecture
        model_input = Input(shape=(self.words_window_size, self.embedding_dims))
        input_reshape = Reshape(target_shape=(self.words_window_size, self.embedding_dims, 1))

        ngram_conv_pool_pair_1_layers = [
            conv_pool_pair(
                num_filters=num_filters,
                kernel_size=(nth_gram, 1),
                conv_activation="tanh",
                # size of nth conv layer output: num_words - nth_gram + 1
                pool_size=(self.words_window_size - nth_gram + 1, self.embedding_dims)
            )
            for nth_gram, num_filters in enumerate(self.ngram_1_filters, 1)
        ]

        min_filter_1_size = min(self.ngram_1_filters)
        ngram_output_1_reshape_layer = Reshape(target_shape=(min_filter_1_size,))
        ngram_output_1_layers = [Dense(units=min_filter_1_size, activation="relu") for _ in range(num_ngrams_1)]
        ngram_output_1_matrix = Reshape(target_shape=(num_ngrams_1, min_filter_1_size, 1))

        ngram_conv_pool_pair_2_layers = [
            conv_pool_pair(
                num_filters=num_filters,
                kernel_size=(nth_gram, 1),
                conv_activation="tanh",
                pool_size=(num_ngrams_1 - nth_gram + 1, min_filter_1_size)
            )
            for nth_gram, num_filters in enumerate(self.ngram_2_filters, 1)
        ]

        min_filter_2_size = min(self.ngram_2_filters)
        ngram_output_2_reshape_layer = Reshape(target_shape=(min_filter_2_size,))
        ngram_output_2_layers = [Dense(units=min_filter_2_size, activation="relu") for _ in range(num_ngrams_2)]
        ngram_output_2_matrix = Reshape(target_shape=(num_ngrams_2, min_filter_2_size, 1))

        conv_pool_pair_3 = conv_pool_pair(
            num_filters=self.conv_3_filters,
            kernel_size=(num_ngrams_2, 1),
            conv_activation="tanh",
            pool_size=(1, min_filter_2_size)
        )

        massage_reshape_layer = Reshape(target_shape=(self.conv_3_filters,))
        massage_layer = Dense(units=self.conv_3_filters, activation="relu")
        output_embedding_layer = Dense(units=self.embedding_dims, activation="relu")

        # ----------- connect layers ----------- #

        new_reshaped_input = input_reshape(model_input)

        ngram_conv_1_layers = [conv_layer(new_reshaped_input) for conv_layer, _ in ngram_conv_pool_pair_1_layers]

        ngram_pool_1_layers = [
            pool_layer(conv_layer)
            for conv_layer, (_, pool_layer) in zip(ngram_conv_1_layers, ngram_conv_pool_pair_1_layers)
        ]

        ngram_dense_1_layers = [
            dense_layer(ngram_output_1_reshape_layer(pool_layer))
            for pool_layer, dense_layer in zip(ngram_pool_1_layers, ngram_output_1_layers)
        ]

        ngram_part_1_output = ngram_output_1_matrix(concatenate(ngram_dense_1_layers))

        ngram_conv_2_layers = [conv_layer(ngram_part_1_output) for conv_layer, _ in ngram_conv_pool_pair_2_layers]

        ngram_pool_2_layers = [
            pool_layer(conv_layer)
            for conv_layer, (_, pool_layer) in zip(ngram_conv_2_layers, ngram_conv_pool_pair_2_layers)
        ]

        ngram_dense_2_layers = [
            dense_layer(ngram_output_2_reshape_layer(pool_layer))
            for pool_layer, dense_layer in zip(ngram_pool_2_layers, ngram_output_2_layers)
        ]

        ngram_part_2_output = ngram_output_2_matrix(concatenate(ngram_dense_2_layers))

        conv_3_layer, pool_3_layer = conv_pool_pair_3
        conv_3_layer = conv_3_layer(ngram_part_2_output)
        pool_3_layer = pool_3_layer(conv_3_layer)
        massage_layer = massage_layer(massage_reshape_layer(pool_3_layer))
        output_embedding_layer = output_embedding_layer(massage_layer)

        return model_input, output_embedding_layer
