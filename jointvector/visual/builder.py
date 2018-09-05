from keras.layers import AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, GlobalAveragePooling2D, Input, ReLU
from keras.models import Model
from keras.regularizers import l2


def add_convolution_block(input_src, nb_filters, kernel_size, strides, bottleneck=False, weight_decay=1e-4):
    bn_layer = BatchNormalization()(input_src)
    relu_layer = ReLU()(bn_layer)

    if bottleneck:
        conv_layer = Conv2D(filters=nb_filters, kernel_size=kernel_size, kernel_regularizer=l2(weight_decay))(relu_layer)
    else:
        conv_layer = Conv2D(filters=nb_filters, kernel_size=kernel_size, strides=strides, padding="same")(relu_layer)

    return conv_layer


class VisualBranchBuilder:
    """
    DenseNet architecture
    """
    def __init__(self, block_sizes, growth_rate, compression_rate, use_bottlenecks=True):
        self.block_sizes = block_sizes
        self.growth_rate = growth_rate
        self.compression_rate = compression_rate
        self.use_bottlenecks = use_bottlenecks

    def build(self):
        nb_channels = 2 * self.growth_rate

        # input
        input_layer = Input(shape=(None, None, 3))

        # initial block
        bottleneck_block_output = add_convolution_block(
            input_src=input_layer,
            nb_filters=nb_channels,
            kernel_size=(7, 7),
            strides=(2, 2),
            bottleneck=True
        )
        avgpool_output = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(bottleneck_block_output)

        # dense blocks
        dense_block_output, transition_block_output = None, None
        for block_size in self.block_sizes[:-1]:
            dense_block_output, nb_channels = self.add_dense_block(avgpool_output, nb_channels, block_size)
            transition_block_output, nb_channels = self.add_transition_block(dense_block_output, nb_channels)

        dense_block_output, nb_channels = self.add_dense_block(transition_block_output, nb_channels, self.block_sizes[-1])

        # classification block
        gap_layer = GlobalAveragePooling2D(data_format="channels_last")(dense_block_output)
        class_output_layer = Dense(units=1000, activation="softmax")(gap_layer)

        return Model(inputs=[input_layer], outputs=[class_output_layer])

    def add_dense_block(self, input_src, nb_channels, block_size):
        updated_input_src = input_src
        feat_maps_list = [input_src]

        for i in range(block_size):
            if self.use_bottlenecks:
                bottleneck_block_output = add_convolution_block(
                    input_src=updated_input_src,
                    nb_filters=4 * self.growth_rate,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    bottleneck=True
                )
                std_block_input = bottleneck_block_output
            else:
                std_block_input = updated_input_src

            std_block_output = add_convolution_block(
                input_src=std_block_input,
                nb_filters=self.growth_rate,
                kernel_size=(3, 3),
                strides=(1, 1)
            )

            feat_maps_list.append(std_block_output)
            updated_input_src = Concatenate()(feat_maps_list)

        updated_nb_channels = nb_channels + self.growth_rate * (block_size - 1)

        return updated_input_src, updated_nb_channels

    def add_transition_block(self, input_src, nb_channels):
        updated_nb_channels = self.compression_rate * nb_channels

        bottleneck_block_output = add_convolution_block(
            input_src=input_src,
            nb_filters=updated_nb_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            bottleneck=True
        )
        avgpool_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bottleneck_block_output)

        return avgpool_output, updated_nb_channels
