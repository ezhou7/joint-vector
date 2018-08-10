from keras.layers import AveragePooling2D, BatchNormalization, Concatenate, Conv2D, ReLU
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
    def __init__(self, growth_rate, compression_rate, use_bottlenecks=True):
        self.growth_rate = growth_rate
        self.compression_rate = compression_rate
        self.use_bottlenecks = use_bottlenecks

    def build(self):
        # TODO: implement this method
        pass

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

        nb_channels += self.growth_rate * (block_size - 1)

        return updated_input_src, nb_channels

    def add_transition_block(self, input_src, nb_channels):
        nb_channels = self.compression_rate * nb_channels

        bottleneck_block_output = add_convolution_block(
            input_src=input_src,
            nb_filters=nb_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            bottleneck=True
        )
        avgpool_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(bottleneck_block_output)

        return avgpool_output, nb_channels
