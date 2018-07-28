import os

from jointvector import ROOT_DIR


def get_props_path():
    return os.path.join(ROOT_DIR, "props/")


def get_task_props_path(props_filename):
    return os.path.join(get_props_path(), props_filename)


class Paths:
    class Resources:
        _dir = "resources/"
        _fasttext_name = "fasttext-50-wikipedia-nytimes-amazon-friends-updated.bin"

        @staticmethod
        def get_fasttext_path():
            return Paths.Resources._dir + Paths.Resources._fasttext_name

    class Transcripts:
        _dir = "data/enhanced-jsons/"
        _transcript_name_template = "friends_season_{0:0>2}.json"
        _num_of_seasons = 4

        @staticmethod
        def get_input_transcript_paths():
            transcript_name_template = Paths.Transcripts._transcript_name_template
            return [
                (
                    Paths.Transcripts._dir + transcript_name_template.format(s + 1),
                    range(1, 20),
                    range(20, 22)
                )
                for s in range(Paths.Transcripts._num_of_seasons)
            ]
