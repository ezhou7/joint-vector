import matplotlib.pyplot as plt
import skimage
from pycocotools.coco import COCO

from jointvector.path import \
    get_train_image_annotations_instances_file_path, \
    get_val_image_annotations_captions_file_path


class CocoDatasetReader:
    def __init__(self):
        self.train_dataset = COCO(get_train_image_annotations_instances_file_path())
        self.val_dataset = COCO(get_val_image_annotations_captions_file_path())

    def get_categories(self):
        pass


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

    image_metadata = coco.loadImgs(ids=[imgIds[0]])[0]
    print(image_metadata)
    img = skimage.io.imread(image_metadata["coco_url"])
    print(img)
    plt.axis("off")
    plt.imshow(img)
    plt.show()
