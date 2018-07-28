import numpy as np

from jointvector.structure import Token


def create_data_split(data, trn_ratio=0.7, dev_ratio=0.15):
    if trn_ratio + dev_ratio > 1.0:
        raise Exception("Sum of ratios cannot exceed 1.0.")
    elif trn_ratio + dev_ratio == 1.0:
        raise Exception("Some data must be used as test data.")

    num_instances = len(data)

    trn = data[:int(num_instances * trn_ratio)]
    dev = data[len(trn):int(num_instances * dev_ratio)]
    tst = data[len(trn) + len(dev):]

    return trn, dev, tst


def generate_features(sentences, word2vec, window_size):
    full_sentences = [sentence.tokens_with_root for sentence in sentences]

    word_vecs_all = [
        [
            word2vec.get_word_vector(token.word_form)
            if not token.is_root_token()
            else Token.get_root_word_vec(word2vec.get_dimension())
            for token in full_sentence
        ]
        for full_sentence in full_sentences
    ]

    half_window = int(window_size / 2)

    features = []
    for word_vecs in word_vecs_all:
        for i, curr_word_vec in enumerate(word_vecs[1:], 1):
            start = 0 if i <= half_window else i - half_window
            end = -1 if len(word_vecs) <= i + half_window else i + half_window

            word_mat = np.array(word_vecs[start:end]).astype("float16")
            num_words = word_mat.shape[0]

            if end - start < window_size:
                full_word_mat = np.zeros((window_size, word2vec.get_dimension())).astype("float16")

                if start == 0:
                    full_word_mat[:num_words] = word_mat
                elif end == -1:
                    full_word_mat[window_size - num_words:] = word_mat
            else:
                full_word_mat = word_mat

            features.append(full_word_mat)

    return np.array(features)
