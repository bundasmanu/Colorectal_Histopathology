WIDTH = 150
HEIGHT = 150
CHANNELS = 3

NUMBER_CLASSES = 8

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

# split percentages of data
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.25
RANDOM_STATE = 0

MULTIPROCESSING = True

X_VAL_ARGS = "X_Val"
Y_VAL_ARGS = "y_val"

PSO_OPTIMIZER = "PSO"
GA_OPTIMIZER = "GA"

ALEX_NET = "ALEXNET"
VGG_NET = "VGGNET"

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
ERROR_ON_SUBSAMPLING = "\n Erro on subsampling, percentage invalid"

#PSO OPTIONS
TOPOLOGY_FLAG = 0
PARTICLES = 1
ITERATIONS = 2
PSO_DIMENSIONS = 5
TOPOLOGY_FLAG = 0 # 0 MEANS GBEST, AND 1 MEANS LBEST
gbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3}
lbestOptions = {'w' : 0.9, 'c1' : 0.3, 'c2' : 0.3, 'k' : 4, 'p' : 2}

#GA OPTIONS
TOURNAMENT_SIZE = 100
INDPB = 0.6
CXPB = 0.4
MUTPB = 0.2
