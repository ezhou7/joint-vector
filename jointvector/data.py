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
            word2vec[token.word_form]
            if not token.is_root()
            else Token.get_root_word_vec(word2vec.shape[1])
            for token in full_sentence.tokens
        ]
        for full_sentence in full_sentences
    ]

    half_window = int(window_size / 2)

    features = [[] for _ in range(window_size)]
    for word_vecs in word_vecs_all:
        for i, curr_word_vec in enumerate(word_vecs[1:], 1):
            start = 0 if i <= half_window else i - half_window
            end = -1 if len(word_vecs) <= i + half_window else i + half_window

            for feature_list, curr_feature in zip(features, word_vecs[start:end]):
                feature_list.append(curr_feature)

    return [np.array(feature_list) for feature_list in features]
