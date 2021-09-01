import os
import numpy as np
import yaml
import warnings
import matplotlib.pyplot as plt
from Data_Generator.DataGenerator import DataGenerator
import cv2
from Training import Trainer
from Model.Main_Model import FcnModel

Config_path = os.path.join(os.curdir, 'Configuration')

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
with open(Config_path) as f:
    config = yaml.load(f)


def read_data(path, is_train=True):
    temp = []
    updated_path = os.path.join(path, "VOC2012_train_val", "ImageSets", "Segmentation",
                                "train.txt" if is_train else "val.txt")
    with open(updated_path, "r") as file_:
        instances = file_.read().split()
        for img in instances:
            path_img = os.path.join(path, "VOC2012_train_val", "JPEGImages", img + ".jpg")
            path_label = os.path.join(path, "VOC2012_train_val", "SegmentationClass", img + ".png")
            temp.append([path_img, path_label])
    return temp


def read_test_data(path):
    temp = []
    updated_path = os.path.join(path, "VOC2012_test", "ImageSets", "Segmentation", "test.txt")
    with open(updated_path, "r") as file_:
        instances = file_.read().split()
        for img in instances:
            path_img = os.path.join(path, "VOC2012_test", "JPEGImages", img + ".jpg")
            temp.append(path_img)
    return temp


path = config["dataset"]
Train = read_data(path=path, is_train=True)
Val = read_data(path=path, is_train=False)
Test = read_test_data(path=path)


def show_some_img(train, images_to_show=5):
    plt.figure(figsize=(15, 15))
    idx = 0
    rows = np.ceil(images_to_show/2)
    for instance in train[:images_to_show]:
        plt.subplot(rows, 2, idx + 1)
        img = cv2.imread(instance[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.subplot(5, 2, idx + 2)
        mask = cv2.imread(instance[1])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        plt.imshow(mask)
        idx += 2


show_some_img(Train, images_to_show=5)
train_generator = DataGenerator(config, Train, shuffle=True, is_train=True, use_aug=config["Data_Aug"]["use_aug"])
Val_generator = DataGenerator(config, Val, shuffle=True, is_train=True, use_aug=config["Data_Aug"]["use_aug"])
test_generator = DataGenerator(config, Test, shuffle=False, is_train=False)


# Loading the Model
model = FcnModel(config, save_model=True, show_summary=False).model()

trainer = Trainer.TrainFcn(config=config, model=model, trainsize=len(Train),
                           Train_Generator=train_generator, Val_Generator=Val_generator)

# Train the model
trainer.train()


