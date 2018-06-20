from keras.layers import Conv2D, MaxPool2D


def conv_pool_pair(num_filters, kernel_size, conv_activation, pool_size):
    conv_layer = Conv2D(filters=num_filters, kernel_size=kernel_size, activation=conv_activation)
    pool_layer = MaxPool2D(pool_size=pool_size)

    return conv_layer, pool_layer
