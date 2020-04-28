import numpy
import config
import os
from glob import glob
import numpy as np
import keras
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from keras.models import Model as mp
from keras.layers import Input, Average
import cv2
from exceptions import CustomError
import pandas as pd

def getImages(directory):

    '''
    THIS FUNTION RETRIEVES ALL IMAGES FILES
    :param directory: str --> dict/*.jpg
    :return: list of all jpg files
    '''

    try:

        return sorted(glob(directory))

    except:
        raise

def addNewColumn_Populate_DataFrame(dataFrame, name_new_column, dataToPopulate):

    '''
    THIS FUNCTION IS USED TO ADD NEW COLUMN TO DATAFRAME, AND POPULATE COLUMN WITH DATA
    :param dataFrame: dataFrame --> dataFrame to apply changes
    :param name_new_column: str --> name of new column
    :param dataToPopulate: List (str) --> strings to populate data
    :return: dataFrame changed
    '''

    try:

        dataFrame[name_new_column] = dataToPopulate
        return dataFrame

    except:
        raise

def get_subsample_of_data(percentage, data):

    '''
    THIS FUNCTION IS USED TO GET SUBSAMPLE OF DATA, CONSIDERED ALWAYS THE INITIAL % OF DATA PER CLASS
    e.g: i want 20% of data
        - i get the same % of samples by all classes:
        e.g: 1000 initial values for class 0 in 2000 samples, in the end i get 200 samples for class 0 (same 50% of samples for this class)
    :param percentage: float --> percentage between 0 and 1, for subsample data
    :param data: DataFrame --> object with all samples
    :return: subsample: DataFrame --> object containg only subsample data
    '''

    try:

        if percentage == 0.0:
            raise CustomError.ErrorCreationModel(config.WARNING_SUBSAMPLING)

        # new subsample
        subsample = pd.DataFrame()

        # get % of rows for each class
        for i in range(config.NUMBER_CLASSES):
            query = "{} == {} ".format(config.TARGET, i)
            sub = data.query(query).sample(frac=percentage, replace=False)
            subsample = subsample.append(sub, ignore_index=True)

        return subsample

    except:
        raise CustomError.ErrorCreationModel(config.ERROR_ON_SUBSAMPLING)

def resize_images(width, height, data):

    '''
    :param width: int --> pixel width to resize image
    :param height: int --> pixel height to resize image
    :param data: dataframe --> shape ["id", "image_path", "target"]
    :return x: numpy array --> shape (number images, width, height)
    :return y: numpy array --> shape (number images, target)
    '''

    try:

        x = []
        y = []

        for i in range(data.shape[0]):
            image = cv2.imread(data.at[i, config.IMAGE_PATH])
            x.append(cv2.resize(image, (width, height)))
            y.append(data.at[i, config.TARGET])

        return numpy.array(x), numpy.array(y)

    except:
        raise

def normalize(X_train, X_val, X_test):

    '''
    #REF https://forums.fast.ai/t/images-normalization/4058/8
    :param X_train: numpy array representing training data
    :param X_val: numpy array representing validation data
    :param X_test: numpy array representing test data
    :return X_train: numpy array normalized
    :return X_val: numpy array normalized
    :return X_X_test: numpy array normalized
    '''

    try:

        mean = np.mean(X_train,axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        std = np.std(X_train, axis=config.STANDARDIZE_AXIS_CHANNELS) #STANDARDIZE BY CHANNELS
        X_train = (X_train-mean)/(std+1e-7)
        X_val = (X_val-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)

        # minmax_scale = preprocessing.MinMaxScaler().fit(X_train)
        # X_train = minmax_scale.transform(X_train)
        # X_val = minmax_scale.transform(X_val)
        # X_test = minmax_scale.transform(X_test)
        #
        # #RESHAPE AGAIN TO 4D
        # shape_data = (X_train.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_train = X_train.reshape(shape_data)
        # shape_data = (X_val.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_val = X_val.reshape(shape_data)
        # shape_data = (X_test.shape[0], config.WIDTH, config.HEIGHT, config.CHANNELS)
        # X_test = X_test.reshape(shape_data)

        return X_train, X_val, X_test
    except:
        raise

def one_hot_encoding(y_train, y_val, y_test):

    '''

    :param y_train: numpy array with training targets
    :param y_val: numpy array with validation targets
    :param y_test: numpy array with test targets
    :return y_train: numpy array categorized [1 0] --> class 0 or [0 1] --> class 1
    :return y_val: numpy array categorized
    :return y_test: numpy array categorized
    '''

    try:

        y_train = keras.utils.to_categorical(y_train, config.NUMBER_CLASSES)
        y_val = keras.utils.to_categorical(y_val, config.NUMBER_CLASSES)
        y_test = keras.utils.to_categorical(y_test, config.NUMBER_CLASSES)

        return y_train, y_val, y_test

    except:
        raise

def decode_array(array):

    '''
    THIS FUNCTION IS USED TO DECODE ENCODING ARRAY'S LIKE PREDICTIONS RESULTED FROM MODEL PREDICT
    e.g : array[[0 1]
                [1 0]]
        return array[[1]
                     [0]]
    :param array: numpy array
    :return: numpy array --> decoded array
    '''

    try:

        decoded_array = np.argmax(array, axis=1) #RETURNS A LIST

        return decoded_array
    except:
        raise

def getConfusionMatrix(predictions, y_test, dict):

    '''
    THIS FUNCTION IS USED IN ORDER TO SHOW MAIN RESULTS OF MODEL EVALUATION (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :param predictions: numpy array --> model predictions
    :param y_test: numpy array --> real targets of test data
    :return: report: dict --> with metrics results (ACCURACY, RECALL, PRECISION OR F-SCORE)
    :return: confusion_mat: ndarray (n_classes, n_classes)
    '''

    try:

        #CREATE REPORT
        if dict == True:
            report = classification_report(y_test, predictions, target_names=config.DICT_TARGETS,
                                       output_dict=True) # returns a dict with metrics to access easily, important in optimizer
        else:
            report = classification_report(y_test, predictions, target_names=config.DICT_TARGETS)
        #CREATION OF CONFUSION MATRIX
        confusion_mat = confusion_matrix(y_test, predictions)

        return report, confusion_mat
    except:
        raise

def plot_cost_history(history):

    '''
    THIS FUNNCTION PLOTS COST HISTORY
    :param history: history object resulted from train
    :return: none --> only plt show
    '''

    try:

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except:
        raise

def plot_accuracy_plot(history):

    '''
    THIS FUNNCTION PLOTS ACCURACY HISTORY
    :param history: history object resulted from train
    :return:
    '''

    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except:
        raise

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def lr_scheduler(epoch):
    return config.LEARNING_RATE * (0.5 ** (epoch // config.DECAY))

def ensemble(models):

    '''
    THIS FUNCTION IS USED TO ENSEMBLE OUTPUT OF A LIST OF MODELS, CONSIDERING ITS AVERAGE
    :param models: List Models : models used (AlexNet, VGGNet, ResNet)
    :return: model: model with average outputs of all models considered
    '''

    try:

        ## get outputs of each model
        input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
        input_model = Input(input_shape)
        models_out = [i(input_model) for i in models]

        ## get average of each model (ensemble)
        average = Average() (models_out)

        ## define model with new outputs

        model = mp(input_model, average, name='ensemble')

        return model

    except:
        raise

def print_final_results(y_test, predictions, history, dict=False):

    '''
    THIS FUNCTION IS USED TO PRINT ANND PLOT FINAL RESULTS OF MODEL EVALUATION
    :param y_test: real predictions of test
    :param predictions: numpy array : predictions of model
    :param history: History.history : history of train (validation and train along epochs)
    :return: nothing only print's and plot's
    '''

    try:

        if history != None:
            print(plot_cost_history(history))
            print(plot_accuracy_plot(history))
        predictions = decode_array(predictions)  # DECODE ONE-HOT ENCODING PREDICTIONS ARRAY
        y_test_decoded = decode_array(y_test)  # DECODE ONE-HOT ENCODING y_test ARRAY
        report, confusion_mat = getConfusionMatrix(predictions, y_test_decoded, dict)
        print(report)
        plt.figure()
        plot_confusion_matrix(confusion_mat, config.DICT_TARGETS)

    except:
        raise

def print_log_message():
    CRED = '\033[91m'
    CBOLD = '\33[1m'
    CEND = '\033[0m'
    print("\n"+CRED+ CBOLD +"Seq: {}".format(next(config.counter_iterations))+CEND)