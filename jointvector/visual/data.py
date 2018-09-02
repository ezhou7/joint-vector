import numpy as np


def search_imagenet(word):
    # TODO: implement this method
    pass


def resize_image(img):
    # TODO: resize image to 256x256
    pass


def sync_with_text(sentences):
    empty_image = np.zeros(shape=(256, 256))

    for sentence in sentences:
        for token in sentence:
            token_image = search_imagenet(token.word_form) if token.pos.startswith("NN") else empty_image
            setattr(token, "image", token_image)
