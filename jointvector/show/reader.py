import json

from jointvector.show import idutils
from jointvector.structure import Token, Sentence
from jointvector.show.transcripts import Utterance, Scene, Episode


def read_season_json(json_path):
    with open(json_path, "r") as fin:
        season_json = json.load(fin)
        episode_jsons = season_json["episodes"]
        episodes = [read_episode_json(episode_json)
                    for episode_json in episode_jsons]

        for i in range(len(episodes) - 1):
            episodes[i + 1].prev_episode = episodes[i]
            episodes[i].next_episode = episodes[i + 1]

        assign_metadata(episodes)

    return episodes


def read_episode_json(episode_json):
    episode_id = episode_json["episode_id"]
    episode_num = idutils.parse_episode_id(episode_id)[-1]

    scene_jsons = episode_json["scenes"]
    scenes = [read_scene_json(scene_json)
              for scene_json in scene_jsons]

    for i in range(len(scenes) - 1):
        scenes[i + 1].prev_scene = scenes[i]
        scenes[i].next_scene = scenes[i + 1]

    return Episode(episode_num, scenes)


def read_scene_json(scene_json):
    scene_id = scene_json["scene_id"]
    scene_num = idutils.parse_scene_id(scene_id)[-1]

    utterance_jsons = scene_json["utterances"]
    utterances = [read_utterance_json(utterance_json) for utterance_json in utterance_jsons]

    for i in range(len(utterances) - 1):
        utterances[i + 1].prev_utterance = utterances[i]
        utterances[i].next_utterance = utterances[i + 1]

    return Scene(scene_num, utterances)


def read_utterance_json(utterance_json):
    speakers = utterance_json["speakers"]

    word_forms = utterance_json["tokens"]
    pos_tags = utterance_json["part_of_speech_tags"]
    dep_tags = utterance_json["dependency_tags"]
    dep_heads = utterance_json["dependency_heads"]
    ner_tags = utterance_json["named_entity_tags"]

    sentences = parse_token_nodes(word_forms, pos_tags, dep_tags, dep_heads, ner_tags)

    return Utterance(speakers, sentences)


def parse_token_nodes(word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
    sentences = []

    # sentence
    for word_s, pos_s, dep_s, h_dep_s, ner_s in zip(word_forms, pos_tags, dep_tags, dep_heads, ner_tags):
        tokens = []

        for idx, word, pos, dep, ner in zip(range(len(word_s)), word_s, pos_s, dep_s, ner_s):
            if pos == "``":
                pos = "''"
            token = Token(word, pos_tag=pos, dep_tag=dep, ner_tag=ner)
            tokens.append(token)

        for idx, hid in enumerate(h_dep_s):
            setattr(tokens[idx], "dep_head", tokens[hid - 1] if hid > 0 else None)

        sentence = Sentence(tokens)
        sentences.append(sentence)

    return sentences


def assign_metadata(episodes):
    for episode in episodes:
        for scene in episode.scenes:
            scene.episode = episode

            for utterance in scene.utterances:
                utterance.scene = scene
