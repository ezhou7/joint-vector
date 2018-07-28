import fastText

from jointvector.model import EmbeddingSystem
from jointvector.path import Paths
from jointvector.task import NLPTask
from jointvector.show.reader import read_season_json


def test_initialization():
    pos_task = NLPTask("pos-task-props.json")

    w2v = fastText.load_model(Paths.Resources.get_fasttext_path())
    assert w2v.get_dimension() == 50

    joint_model = EmbeddingSystem([pos_task], w2v)
    assert len(joint_model.models) == 1


def test_training():
    pos_task = NLPTask("pos-task-props.json")
    w2v = fastText.load_model(Paths.Resources.get_fasttext_path())
    joint_model = EmbeddingSystem([pos_task], w2v)

    transcript_paths = Paths.Transcripts.get_input_transcript_paths()
    first_season_transcript = transcript_paths[0]
    episodes = read_season_json(first_season_transcript[0])
    sentences = [
        sentence
        for episode in episodes
        for scene in episode.scenes
        for utterance in scene.utterances
        for sentence in utterance.sentences
    ]

    joint_model.train(sentences, sentences)
