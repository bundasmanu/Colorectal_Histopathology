import numpy as np
import itertools

# global counter
counter_iterations = itertools.count(start=0, step=1)

# image dimensions
WIDTH = 150
HEIGHT = 150
CHANNELS = 3

NUMBER_CLASSES = 8
STANDARDIZE_AXIS_CHANNELS = (0,1,2,3)

# directories to get images
INPUT_DIR = 'input'
STROMA_DIR = 'STROMA'
TUMOR_DIR = 'TUMOR'
ADIPOSE_DIR = 'ADIPOSE'
COMPLEX_DIR = 'COMPLEX'
DEBRIS_DIR = 'DEBRIS'
EMPTY_DIR = 'EMPTY'
LYMPHO_DIR = 'LYMPHO'
MUCOSA_DIR = 'MUCOSA'
IMAGES_ACESS = 'images'
IMAGES_REGEX = '*.tif'

# columns of DataFrame
IMAGE_PATH = 'image_path'
TARGET = 'target'

# SUBSAMPLE PERCENTAGE
SUBSAMPLE_PERCENTAGE = 1

# split percentages of data
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.25
RANDOM_STATE = 0

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

# model type str
ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"
RES_NET = "RESNET"

# activation functions
RELU_FUNCTION = "relu"
SOFTMAX_FUNCTION = "softmax"
SIGMOID_FUNCTION = "sigmoid"

# padding types
VALID_PADDING = "valid"
SAME_PADDING = "same"

# regularization and train optimizer parameters
LEARNING_RATE = 0.001
DECAY = 1e-6

# train function's of loss
LOSS_BINARY = "binary_crossentropy"
LOSS_CATEGORICAL = "categorical_crossentropy"

# train metrics
ACCURACY_METRIC = "accuracy"
VALIDATION_ACCURACY = "val_accuracy"
LOSS = "loss"
VALIDATION_LOSS = "val_loss"

# optimizer type str
PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

# train parameters
BATCH_SIZE_ALEX_NO_AUG = 16
BATCH_SIZE_ALEX_AUG = 16
EPOCHS = 20
MULTIPROCESSING = True
SHUFFLE = True
GLOROT_SEED = 0

# data augmentation options
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ROTATION_RANGE = 10
ZOOM_RANGE = 0.25
BRITNESS_RANGE= 0.3

#EXCEPTIONS MESSAGES
ERROR_MODEL_EXECUTION = "\nError on model execution"
ERROR_NO_ARGS = "\nPlease provide args: ",X_VAL_ARGS," and ", Y_VAL_ARGS
ERROR_NO_ARGS_ACCEPTED = "\nThis Strategy doesn't accept more arguments"
ERROR_NO_MODEL = "\nPlease pass a initialized model"
ERROR_INVALID_OPTIMIZER = "\nPlease define a valid optimizer: ", PSO_OPTIMIZER," or ", GA_OPTIMIZER
ERROR_INCOHERENT_STRATEGY = "\nYou cannot choose the oversampling and undersampling strategies at the same time"
ERROR_ON_UNDERSAMPLING = "\nError on undersampling definition"
ERROR_ON_OVERSAMPLING = "\nError on oversampling definition"
ERROR_ON_DATA_AUG = "\nError on data augmentation definition"
ERROR_ON_TRAINING = "\nError on training"
ERROR_ON_OPTIMIZATION = "\nError on optimization"
ERROR_INVALID_NUMBER_ARGS = "\nPlease provide correct number of args"
ERROR_ON_BUILD = "\nError on building model"
ERROR_APPEND_STRATEGY = "\nError on appending strategy"
ERROR_ON_PLOTTING = "\nError on plotting"
ERROR_ON_SUBSAMPLING = "\n Error on subsampling, percentage invalid"
WARNING_SUBSAMPLING = "\nIf you want to subsampling data, please pass a value >0 and <1"

#PSO OPTIONS
PARTICLES = 2
ITERATIONS = 2
PSO_DIMENSIONS = 5
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3}
lbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3, 'k' : 4, 'p' : 2} # p =2, means euclidean distance

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2

# dict with target's name, for print confusion matrix
DICT_TARGETS = (
    'STROMA' ,
    'TUMOR' ,
    'MUCOSA' ,
    'EMPTY' ,
    'LYMPHO' ,
    'ADIPOSE',
    'COMPLEX',
    'DEBRIS'
)

# classes weights
class_weights={
    0: 3.0, # stroma
    1: 1.2, # tumor
    2: 1.2, # mucosa
    3: 1.0, # empty
    4: 1.2, # lympho
    5: 1.0, # adipose
    6: 3.0, # complex
    7: 2.0  # debris
}

# PSO BOUNDS LIMITS --> (needs to be readjusted, in coherence with this specific problem, and with computational costs)
MAX_VALUES_LAYERS_ALEX_NET = [196, 196, 196, 256, 256, 128, 80] # 6 convs, 1 dense and batch size
MAX_VALUES_LAYERS_VGG_NET = [196, 196, 196, 196, 256, 128, 80] # 5 convs, 1 dense and batch_size
MAX_VALUES_LAYERS_RESNET_NET = [128, 128, 128, 196, 196, 196, 196, 256, 256, 80] ## 8 convs and batch size
MIN_BATCH_SIZE = 16

# weights model files
VGG_NET_WEIGHTS_FILE = 'vggnet_weights.h5'
ALEX_NET_WEIGHTS_FILE = 'alexnet_weights.h5'
RES_NET_WEIGHTS_FILE = 'resnet_weights.h5'

# filename position particles along iterations (.html file)
PSO_POSITION_ITERS = 'pos_iters.html'

# variables that are used in plot positions of particles along iterations
X_LIMITS = [1, 128]
Y_LIMITS = [1, 128]
LABEL_X_AXIS = 'Nºfiltros 1ªcamada'
LABEL_Y_AXIS = 'Nºfiltros 2ªcamada'

# PSO INIT DEFINITIONS --> IN ARGS FORM
pso_init_args_alex = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    7,  # dimensions (5 conv filters, 1 dense neurons and batch size)
    np.array(MAX_VALUES_LAYERS_ALEX_NET)  # superior bound limits for dimensions
)

pso_init_args_vgg = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    7,  # dimensions (5 conv filters, 1 dense neurons and batch size)
    np.array(MAX_VALUES_LAYERS_VGG_NET)  # superior bound limits for dimensions
)

pso_init_args_resnet = (
    PARTICLES,  # number of individuals
    ITERATIONS,  # iterations
    10,  # dimensions (9 conv filters and batch size)
    np.array(MAX_VALUES_LAYERS_RESNET_NET)  # superior bound limits for dimensions
)