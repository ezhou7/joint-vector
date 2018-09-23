import cv2
import numpy as np
from pycocotools.coco import COCO
from typing import Set, List, Tuple

from jointvector.path import \
    get_train_image_annotations_instances_file_path, \
    get_val_image_annotations_instances_file_path, \
    get_train_image, \
    get_val_image


class CocoDataset:
    def __init__(self):
        self.train_split = COCO(get_train_image_annotations_instances_file_path())
        self.val_split = COCO(get_val_image_annotations_instances_file_path())

        self.index_to_category = {i: self.train_split.cats[i] for i in range(len(self.train_split.cats))}

    def get_categories(self) -> Set[str]:
        return set(category["name"] for category in self.train_split.cats)

    def get_category_from_index(self, index) -> str:
        return self.index_to_category[index]

    def get_supercategories(self) -> Set[str]:
        return set(category["supercategory"] for category in self.train_split.cats)

    def get_train_image_file_names(self) -> List[str]:
        return [image_metadata["file_name"] for image_metadata in self.train_split.imgs]

    def get_train_images(self) -> List[np.array]:
        image_file_names = self.get_train_image_file_names()
        return [cv2.imread(get_train_image(image_file_name)) for image_file_name in image_file_names]

    def get_val_image_file_names(self) -> List[str]:
        return [image_metadata["file_name"] for image_metadata in self.val_split.imgs]

    def get_val_images(self) -> List[np.array]:
        image_file_names = self.get_val_image_file_names()
        return [cv2.imread(get_val_image(image_file_name)) for image_file_name in image_file_names]

    def get_train_instances(self) -> Tuple[List[np.array], List[List[dict]], List[int]]:
        images = self.get_train_images()
        image_annotations = [self.train_split.imgToAnns[image_id] for image_id in self.train_split.imgs]
        image_category_labels = [image_annotation["category_id"] for image_annotation in image_annotations]

        return images, image_annotations, image_category_labels

    def get_val_instances(self):
        images = self.get_val_images()
        image_annotations = [self.train_split.imgToAnns[image_id] for image_id in self.val_split.imgs]
        image_category_labels = [image_annotation["category_id"] for image_annotation in image_annotations]

        return images, image_annotations, image_category_labels


if __name__ == "__main__":
    annotation_path = "/Users/ezhou7/Downloads/coco-dataset/annotations/instances_val2017.json"

    coco = COCO(annotation_path)
    cats = coco.loadCats(coco.getCatIds())
    category_names = [cat["name"] for cat in cats]
    print("COCO categories: \n{}\n".format(" ".join(category_names)))

    supercategory_names = set(cat["supercategory"] for cat in cats)
    print("COCO supercategories: \n{}\n".format(" ".join(supercategory_names)))

    catIds = coco.getCatIds(catNms=["person", "dog", "skateboard"])
    imgIds = coco.getImgIds(catIds=catIds)
    print(catIds)
    print(imgIds)

    image_metadata = coco.loadImgs(ids=[imgIds[0]])[0]
    print(image_metadata.keys())
    print(image_metadata["file_name"])

    images_metadata = coco.loadImgs(ids=imgIds)
    print([(metadata["height"], metadata["width"]) for metadata in images_metadata])
    print(coco.anns.keys())
