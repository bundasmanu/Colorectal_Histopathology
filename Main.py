from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import pandas as pd
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
    IMAGE_REGEX = os.path.join(INPUT_DIR, config.IMAGES_ACESS) # regex path for all images on images directory

    # get csv data file with train labels, and convert it to DataFrame object
    CSV_FILE_PATH = os.path.join(INPUT_DIR, config.TRAIN_LABELS_CSV)
    data = pd.read_csv(CSV_FILE_PATH)
    print(data.head(5))

    # get all images paths in coherent order with DataFrame
    images_files = config_func.getImages(IMAGE_REGEX) # glob retrieves data in this directory
    print(images_files[:5])

    # add image path column to DataFrame, associated with it's respective column
    data = config_func.addNewColumn_Populate_DataFrame(data,config.IMAGE_PATH, images_files)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    print(data.info())
    print(data.head(5))

    # subsample data, if if not wanted, the rate 1 should be passed
    data = config_func.get_subsample_of_data(1, data)
    print(data.head(5))

    return None

if __name__ == "__main__":
    main()