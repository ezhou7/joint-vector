import os

from jointvector import ROOT_DIR


# ---- property/configuration paths ---- #

def get_props_dir_path():
    return os.path.join(ROOT_DIR, "props/")


def get_task_props_file_path(props_filename):
    return os.path.join(get_props_dir_path(), props_filename)


# ---- resources path ---- #

def get_resources_dir_path():
    return os.path.join(ROOT_DIR, "resources/")


# ---- word embedding path ---- #

def get_fasttext_file_path():
    return os.path.join(get_resources_dir_path(), "fasttext-50-wikipedia-nytimes-amazon-friends-updated.bin")


# ---- image dataset paths ---- #

def get_image_dataset_dir_path():
    return os.path.join(get_resources_dir_path(), "coco-dataset/")


def get_images_dir_path():
    return os.path.join(get_image_dataset_dir_path(), "images/")


def get_train_images_dir_path():
    return os.path.join(get_images_dir_path(), "train2017/")


def get_val_images_dir_path():
    return os.path.join(get_images_dir_path(), "val2017/")


def get_test_images_dir_path():
    return os.path.join(get_images_dir_path(), "test2017/")


# ---- image dataset annotations paths ---- #

def get_image_annotations_dir_path():
    return os.path.join(get_image_dataset_dir_path(), "annotations/")


def get_train_image_annotations_captions_file_path():
    return os.path.join(get_image_annotations_dir_path(), "captions_train2017.json")


def get_val_image_annotations_captions_file_path():
    return os.path.join(get_image_annotations_dir_path(), "captions_val2017.json")


def get_train_image_annotations_instances_file_path():
    return os.path.join(get_image_annotations_dir_path(), "instances_train2017.json")


def get_val_image_annotations_instances_file_path():
    return os.path.join(get_image_annotations_dir_path(), "instances_val2017.json")


def get_train_image_annotations_person_keypoints_file_path():
    return os.path.join(get_image_annotations_dir_path(), "person_keypoints_train2017.json")


def get_val_image_annotations_person_keypoints_file_path():
    return os.path.join(get_image_annotations_dir_path(), "person_keypoints_val2017.json")


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
