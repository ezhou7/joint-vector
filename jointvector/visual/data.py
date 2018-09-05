import numpy as np


def search_images(word):
    # TODO: implement this method
    pass


def sync_with_text(sentences):
    empty_image = np.zeros(shape=(32, 32))

    for sentence in sentences:
        for token in sentence:
            token_image = search_images(token.word_form) if token.pos.startswith("NN") else empty_image
            setattr(token, "image", token_image)
