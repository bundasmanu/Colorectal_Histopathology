from . import Optimizer
from models import Model, DenseNet
from exceptions import CustomError
import config
import pyswarms as ps
import numpy as np
from typing import Tuple
import math
from pyswarms.utils.plotters import plot_cost_history, plot_contour
import matplotlib.pyplot as plt
import config_func
from pyswarms.utils.plotters.formatters import Designer
from keras import backend as K
import gc

class PSO(Optimizer.Optimizer):

    def __init__(self, model : Model.Model, *args): #DIMENSIONS NEED TO BE EQUAL TO NUMBER OF LAYERS ON MODEL
        '''
        :param args:
            - individuals (Parent attribute): integer --> number of particles
            - dimensions (Parent attribute): integer --> number dimensions of problems (number hyperparameters to optimize)
            - iterations (Parent attribute): integer --> number of iterations
            - limit_super (PSO class attribute): numpy array --> array with max values for each dimension of problem (dimensions, )
                * if user needs to use lower_limit for dimensions different from 1,
                    needs to override __init__, and after that needs to override boundsDefinition method
        '''
        self.limit_infer = args[-2]
        self.limit_super = args[-1]  # last argument
        super(PSO, self).__init__(model, *args[:-2]) # all args except last one --> last one is limit_super, that is a attribute of PSO concrete class

    def plotCostHistory(self, optimizer):

        '''
        :param optimizer: optimizer object returned in the application/definition of PSO
        '''

        try:

            plot_cost_history(cost_history=optimizer.cost_history)

            plt.show()

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_PLOTTING)

    def plotPositionHistory(self, optimizer, xLimits, yLimits, filename, xLabel, yLabel):

        '''
        :param optimizer: optimizer object returned in the application/definition of PSO
        :param xLimits: numpy array (minLimit, maxLimit) of x Axis
        :param yLimits: numpy array (minLimit, maxLimit) of y Axis
        :param filename: name of filename returned by plot_contour (html gif)
        :param xLabel: name of X axis
        :param yLabel: name of Y axis
        '''

        try:

            d = Designer(limits=[xLimits, yLimits], label=[xLabel, yLabel])
            pos = []
            for i in range(config.ITERATIONS):
                pos.append(optimizer.pos_history[i][:, 0:2])
            animation = plot_contour(pos_history=pos,
                                     designer=d)

            plt.close(animation._fig)
            html_file = animation.to_jshtml()
            with open(filename, 'w') as f:
                f.write(html_file)

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_PLOTTING)

    def boundsDefinition(self):

        '''
        This function has as main objective to define the limits of the dimensions of the problem
        :return: 2 numpy arrays --> shape(dimensionsProblem, ) with min and max values for each dimension of the problem
        '''

        try:

            totalDimensions = self.dims

            minBounds = np.ones(totalDimensions)
            minBounds = [minBounds[j] * i for i, j in zip(self.limit_infer, range(totalDimensions))]
            minBounds = np.array(minBounds)

            maxBounds = np.ones(totalDimensions)
            maxBounds = [maxBounds[j] * i for i, j in zip(self.limit_super, range(totalDimensions))]
            maxBounds = np.array(maxBounds)

            bounds = (minBounds, maxBounds)

            return bounds

        except:
            raise

    def objectiveFunction(self, acc, *args):

        '''
        Concrete objective function of PSO object --> can override or not Parent function
        :param acc: float --> model accuracy
        :param args: first argument is a Keras Model
                    last argument is a confusion matrix
                    * if user needs can pass more arguments
        :return: float --> particle cost, on a given iteration
        '''

        return super(PSO, self).objectiveFunction(acc, *args)

    def loopAllParticles(self, particles):

        '''
        THIS FUNCTION APPLIES PARTICLES ITERATION, EXECUTION CNN MODEL
        :param particles: numpy array of shape (nParticles, dimensions)
        :return: list of floats: costs of particles in a given iteration (nParticles, )
        '''

        try:

            losses = []
            for i in range(particles.shape[0]):
                config_func.print_log_message()
                if isinstance(self.model, DenseNet.DenseNet) == True:
                    int_converted_values = [math.trunc(j) for j in particles[i][:-2]]
                    int_converted_values.append(particles[i][-2])  # compression rate --> float
                    int_converted_values.append(math.trunc(particles[i][-1]))
                else:
                    int_converted_values = [math.trunc(i) for i in particles[i]]  # CONVERSION OF DIMENSION VALUES OF PARTICLE
                print(int_converted_values)
                model, predictions, history = self.model.template_method(*int_converted_values) #APPLY BUILD, TRAIN AND PREDICT MODEL OPERATIONS, FOR EACH PARTICLE AND ITERATION
                decoded_predictions = config_func.decode_array(predictions)
                decoded_y_true = config_func.decode_array(self.model.data.y_test)
                report, conf = config_func.getConfusionMatrix(decoded_predictions, decoded_y_true, dict=True)
                acc = report['accuracy']# i can't compare y_test and predict, because some classes may have been unclassified
                # define args to pass to objective function
                obj_args = (model, report)
                losses.append(self.objectiveFunction(acc, *obj_args)) #ADD COST LOSS TO LIST
                K.clear_session()
                gc.collect()
                del model
            return losses

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)

    def optimize(self) -> Tuple[float, float, ps.single.general_optimizer.SwarmOptimizer]:

        '''
        THIS FUNCTION IS RESPONSIBLE TO APPLY ALL LOGIC OF PSO CNN NETWORK OPTIMIZATION
        :return: [float, float, SwarmOptimizer] --> best cost, best particle position for each dimension and SwarmOptimizer (Pyswarms object)
        '''

        try:

            #DEFINITION OF BOUNDS
            bounds = self.boundsDefinition()

            optimizer = None
            if config.TOPOLOGY_FLAG == 0: #global best topology
                optimizer = ps.single.GlobalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.gbestOptions, bounds=bounds, bh_strategy='shrink', vh_strategy='invert')
            else: #local best topology
                optimizer = ps.single.LocalBestPSO(n_particles=self.indiv, dimensions=self.dims,
                                                    options=config.lbestOptions, bounds=bounds, bh_strategy='shrink', vh_strategy='invert')

            cost, pos = optimizer.optimize(objective_func=self.loopAllParticles, iters=self.iters)
            return cost, pos, optimizer

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_OPTIMIZATION)