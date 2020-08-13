from abc import ABC, abstractmethod
from models import Model
from exceptions import CustomError
import config
import numpy as np
from keras import backend as K

class Optimizer(ABC):

    def __init__(self, model : Model.Model, individuals, iterations, dimensions):
        if model == None:
            raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)
        self.model = model
        self.indiv = individuals
        self.iters = iterations
        self.dims = dimensions

    @abstractmethod
    def objectiveFunction(self, acc, *args):

        '''
        THIS FUNCTION REPRESENTS PARENT OBJECTIVE FUNCTION OF OPTIMIZER
            - IF USER WANTS CAN OVERRIDE THIS ON CONCRETE OPTIMIZER CLASS METHOD
        :param acc: final score on train
        :param args: first argument is a Keras Model
                    last argument is a confusion matrix
        :return: cost : float
        '''

        try:

            # get report
            report = args[-1]
            #recall_idc = report['macro avg']['recall']
            #precision_idc = report['macro avg']['precision']
            f1_idc = report['macro avg']['f1-score']

            # get model
            model = args[0]
            trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])

            return 1e-9 * trainable_count + 6.0 * (1.0 - f1_idc)

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)

    @abstractmethod
    def optimize(self):
        pass