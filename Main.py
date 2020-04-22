from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config
import config_func
import os

def main():

    print("\n###############################################################")
    print("##########################DATA PREPARATION#####################")
    print("###############################################################\n")

    # acess image data
    PROJECT_DIR = os.getcwd()
    INPUT_DIR = os.path.join(PROJECT_DIR, config.INPUT_DIR) # path of input directory
    IMAGES_DIR = os.path.join(INPUT_DIR, config.IMAGES_ACESS)

    # define paths for all classes (stroma, tumor, mucosa, empty, lympho, adipose, complex, debris)
    STROMA_FOLDER = os.path.join(IMAGES_DIR, config.STROMA_DIR, config.IMAGES_REGEX)
    TUMOR_FOLDER = os.path.join(IMAGES_DIR, config.TUMOR_DIR, config.IMAGES_REGEX)
    MUCOSA_FOLDER = os.path.join(IMAGES_DIR, config.MUCOSA_DIR, config.IMAGES_REGEX)
    EMPTY_FOLDER = os.path.join(IMAGES_DIR, config.EMPTY_DIR, config.IMAGES_REGEX)
    LYMPHO_FOLDER = os.path.join(IMAGES_DIR, config.LYMPHO_DIR, config.IMAGES_REGEX)
    ADIPOSE_FOLDER = os.path.join(IMAGES_DIR, config.ADIPOSE_DIR, config.IMAGES_REGEX)
    COMPLEX_FOLDER = os.path.join(IMAGES_DIR, config.COMPLEX_DIR, config.IMAGES_REGEX)
    DEBRIS_FOLDER = os.path.join(IMAGES_DIR, config.DEBRIS_DIR, config.IMAGES_REGEX)
    LIST_CLASSES_FOLDER = [
        STROMA_FOLDER, TUMOR_FOLDER, MUCOSA_FOLDER, EMPTY_FOLDER,
        LYMPHO_FOLDER, ADIPOSE_FOLDER, COMPLEX_FOLDER, DEBRIS_FOLDER
    ]

    # get images from all folders
    # classes targets --> 0: Stroma, 1: Tumor, 2: Mucosa, 3: Empty, 4: Lympho, 5: Adipose, 6: Complex, 7: Debris
    images = []
    labels = []
    for i, j in zip(LIST_CLASSES_FOLDER, range(config.NUMBER_CLASSES)):
        images.append(config_func.getImages(i))
        labels.extend([j for i in range(len(images[j]))])

    # flatten images list
    images = [path for sublist in images for path in sublist]

    # construct DataFrame with two columns: (image_path, target)
    data = pd.DataFrame(
        list(zip(images, labels))
        ,columns=[config.IMAGE_PATH, config.TARGET])

    # subsample data, if not wanted, rate 1 should be passed
    if config.SUBSAMPLE_PERCENTAGE != 1:
        data = config_func.get_subsample_of_data(1, data)
        print(data.head(5))
        print(data.shape)
        print(data[config.TARGET].value_counts())

    # get pixel data from images and respectives targets
    X, Y = config_func.resize_images(config.WIDTH, config.HEIGHT, data)
    print(X.shape)
    print(Y.shape)

    # STRATIFY X_TEST, X_VAL AND X_TEST
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=config.VALIDATION_SPLIT, shuffle=True,
                                                      random_state=config.RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=config.TEST_SPLIT,
                                                        shuffle=True, random_state=config.RANDOM_STATE)

    print(X_train.shape)
    print(pd.value_counts(y_train))
    print(X_val.shape)
    print(X_test.shape)

    return None

if __name__ == "__main__":
    main()