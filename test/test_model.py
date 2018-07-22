import fastText

from jointvector.model import EmbeddingSystem
from jointvector.task import NLPTask


def test_initialization():
    pos_task = NLPTask("pos-task-props.json")
    w2v = fastText.load_model("resources/fasttext-50-wikipedia-nytimes-amazon-friends-180614.bin")
    joint_model = EmbeddingSystem([pos_task], w2v)
    assert len(joint_model.models) == 1
    assert len(joint_model.word2vec[0]) == 50
