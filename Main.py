from exceptions import CustomError
from models import AlexNet, VGGNet, Model, ModelFactory
from models.Strategies_Train import DataAugmentation, Strategy, UnderSampling, OverSampling
from optimizers import GA, PSO, Optimizer, OptimizerFactory
import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
import config
import config_func
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #THIS LINE DISABLES GPU OPTIMIZATION

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

    # normalization of data
    X_train, X_val, X_test = config_func.normalize(X_train, X_val, X_test)

    # one-hot encoding targets
    y_train, y_val, y_test = config_func.one_hot_encoding(y_train=y_train, y_val=y_val, y_test=y_test)

    print("\n###############################################################")
    print("##########################CLASSIFICATION#######################")
    print("###############################################################\n")

    # creation of Data instance
    data_obj = Data.Data(X_train=X_train, X_val=X_val, X_test=X_test,
                         y_train=y_train, y_val=y_val, y_test=y_test)

    # creation of Factory model's instance
    model_factory = ModelFactory.ModelFactory()

    # creation of Factory optimization algorithms instance
    optimization_factory = OptimizerFactory.OptimizerFactory()

    # definition of train strategies instances
    undersampling = UnderSampling.UnderSampling()
    oversampling = OverSampling.OverSampling()
    data_augment = DataAugmentation.DataAugmentation()

    ## ---------------------------ALEXNET APPLICATION ------------------------------------

    # number of conv layers and dense respectively
    alex_number_layers = (
        5,
        1
    )

    # creation of AlexNet instance
    alexNet = model_factory.getModel(config.ALEX_NET, data_obj, *alex_number_layers)

    # apply strategies to alexNet
    alexNet.addStrategy(data_augment)

    # definition of args to pass to template_method (conv's number of filters, dense neurons and batch size)
    alex_args = (
        2, # number of normal convolutional layer (+init conv)
        3, # number of stack cnn layers
        16, # number of feature maps of initial conv layer
        16, # growth rate
        1, # number of FCL Layers
        16, # number neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG # batch size
    )

    # apply build, train and predict
    #model, predictions, history = alexNet.template_method(*alex_args)
    ##alexNet.save(model, config.ALEX_NET_WEIGHTS_FILE)

    # print final results
    '''config_func.print_final_results(y_test=data_obj.y_test, predictions=predictions,
                                    history=history, dict=False)'''

    ## ---------------------------VGGNET APPLICATION ------------------------------------

    # number of conv layers and dense respectively
    vgg_number_layers = (
        4,
        1
    )

    # creation of VGGNet instance
    vggnet = model_factory.getModel(config.VGG_NET, data_obj, *vgg_number_layers)

    # apply strategies to vggnet
    vggnet.addStrategy(data_augment)

    # definition of args to pass to template_method (conv's number of filters, dense neurons and batch size)

    vgg_args = (
        4,  # number of stack cnn layers (+ init stack)
        64,  # number of feature maps of initial conv layer
        12,  # growth rate
        1, # number of FCL Layers
        16,  # number neurons of Full Connected Layer
        config.BATCH_SIZE_ALEX_AUG  # batch size
    )

    # apply build, train and predict
    #model, predictions, history = vggnet.template_method(*vgg_args)
    ##vggnet.save(model, config.VGG_NET_WEIGHTS_FILE)

    # print final results
    '''config_func.print_final_results(y_test=data_obj.y_test, predictions=predictions,
                                    history=history, dict=False)'''

    ## ---------------------------RESNET APPLICATION ------------------------------------

    # number of conv and dense layers respectively
    number_cnn_dense = (5, 1)

    # creation of ResNet instance
    resnet = model_factory.getModel(config.RES_NET, data_obj, *number_cnn_dense)

    # apply strategies to resnet
    resnet.addStrategy(data_augment)

    # definition of args to pass to template_method (conv's number of filters, dense neurons and batch size)
    resnet_args = (
        48, # number of filters of initial CNN layer
        4, # number of consecutive conv+identity blocks
        8, # growth rate
        config.BATCH_SIZE_ALEX_AUG, # batch size
    )

    # apply build, train and predict
    model, predictions, history = resnet.template_method(*resnet_args)
    ##resnet.save(model, config.RES_NET_WEIGHTS_FILE)

    # print final results
    config_func.print_final_results(y_test=data_obj.y_test, predictions=predictions,
                                    history=history, dict=False)

    ## --------------------------- ENSEMBLE OF MODELS ------------------------------------

    # get weights of all methods from files
    # alexNet = load_model(config.ALEX_NET_WEIGHTS_FILE)
    # vggnet = load_model(config.VGG_NET_WEIGHTS_FILE)
    # resnet = load_model(config.RES_NET_WEIGHTS_FILE)
    #
    # models = [alexNet, vggnet, resnet]
    #
    # ##call ensemble method
    # ensemble_model = config_func.ensemble(models=models)
    # predictions = ensemble_model.predict(data_obj.X_test)
    # argmax_preds = np.argmax(predictions, axis=1)  # BY ROW, BY EACH SAMPLE
    # argmax_preds = keras.utils.to_categorical(argmax_preds)
    #
    # ## print final results
    # config_func.print_final_results(data_obj.y_test, argmax_preds, history=None, dict=False)

    ## --------------------------- PSO ------------------------------------------------

    # optimizer fabric object
    # opt_fact = OptimizerFactory.OptimizerFactory()
    #
    # # definition models optimizers
    # pso_alex = opt_fact.createOptimizer(config.PSO_OPTIMIZER, alexNet, *config.pso_init_args_alex)
    # pso_vgg = opt_fact.createOptimizer(config.PSO_OPTIMIZER, vggnet, *config.pso_init_args_vgg)
    # pso_resnet = opt_fact.createOptimizer(config.PSO_OPTIMIZER, resnet, *config.pso_init_args_resnet)
    #
    # # optimize and print best cost
    # cost, pos, optimizer = pso_alex.optimize()
    # print(cost)
    # print(pos)
    # pso_alex.plotCostHistory(optimizer)
    # pso_alex.plotPositionHistory(optimizer, np.array(config.X_LIMITS), np.array(config.Y_LIMITS), config.PSO_POSITION_ITERS,
    #                            config.LABEL_X_AXIS, config.LABEL_Y_AXIS)

if __name__ == "__main__":
    main()