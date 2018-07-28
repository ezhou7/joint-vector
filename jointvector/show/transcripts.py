class Episode:
    def __init__(self, id_num, scenes, prev_episode=None, next_episode=None):
        self.id = int(id_num)
        self.scenes = scenes if scenes is not None else []

        self.prev_episode = prev_episode
        self.next_episode = next_episode


class Scene:
    def __init__(self, id_num, utterances, episode=None, prev_scene=None, next_scene=None):
        self.id = int(id_num)
        self.utterances = utterances if utterances is not None else []

        self.episode = episode

        self.prev_scene = prev_scene
        self.next_scene = next_scene


class Utterance:
    def __init__(self, speakers, sentences, scene=None, prev_utterance=None, next_utterance=None):
        self.speakers = speakers
        self.sentences = sentences if sentences is not None else []

        self.scene = scene

        self.prev_utterance = prev_utterance
        self.next_utterance = next_utterance
