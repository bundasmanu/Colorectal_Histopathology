from . import Model
import Data
from exceptions import CustomError
import config
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from keras.callbacks.callbacks import History
from typing import Tuple
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils import class_weight
import config_func
import numpy
from .Strategies_Train import Strategy, DataAugmentation
from keras import regularizers

class VGGNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(VGGNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(VGGNet, self).addStrategy(strategy)

    def build(self, *args, trainedModel=None) -> Sequential:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR THE INITIALIZATION OF SEQUENTIAL ALEXNET MODEL
        :param args: list integers, in logical order --> to populate cnn (filters) and dense (neurons)
        :return: Sequential: AlexNet MODEL
        '''

        try:

            #IF USER ALREADY HAVE A TRAINED MODEL, AND NO WANTS TO BUILD AGAIN A NEW MODEL
            if trainedModel != None:
                return trainedModel

            if len(args) < (self.nDenseLayers+self.nCNNLayers):
                raise CustomError.ErrorCreationModel(config.ERROR_INVALID_NUMBER_ARGS)

            model = Sequential()

            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            model.add(Conv2D(filters=args[0], input_shape=input_shape, kernel_size=(5, 5),
                            kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[0], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=args[1], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[1], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=args[2], kernel_size=(3, 3),
                            kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[2], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=args[3], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[3], kernel_size=(3, 3),
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))

            model.add(Conv2D(filters=args[4], kernel_size=(3, 3), padding=config.SAME_PADDING,
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(Conv2D(filters=args[4], kernel_size=(3, 3), padding=config.SAME_PADDING,
                             kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
            model.add(BatchNormalization())
            model.add(Dropout(0.25))

            model.add(Flatten())

            model.add(Dense(units=args[5], kernel_regularizer=regularizers.l2(config.DECAY)))
            model.add(Activation(config.RELU_FUNCTION))
            model.add(BatchNormalization())

            model.add(Dense(units=config.NUMBER_CLASSES))
            model.add(Activation(config.SOFTMAX_FUNCTION))

            if config.BUILD_SUMMARY == 1:
                model.summary()

            return model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_BUILD)

    def train(self, model : Sequential, *args) -> Tuple[History, Sequential]:

        '''
        THIS FUNCTION IS RESPONSIBLE FOR MAKE THE TRAINING OF MODEL
        :param model: Sequential model builded before, or passed (already trained model)
        :return: Sequential model --> trained model
        :return: History.history --> train and validation loss and metrics variation along epochs
        '''

        try:

            if model is None:
                raise CustomError.ErrorCreationModel(config.ERROR_NO_MODEL)

            # OPTIMIZER
            opt = Adam(learning_rate=config.LEARNING_RATE, decay=config.DECAY)

            # COMPILE
            model.compile(optimizer=opt, loss=config.LOSS_CATEGORICAL, metrics=[config.ACCURACY_METRIC])

            # GET STRATEGIES RETURN DATA, AND IF DATA_AUGMENTATION IS APPLIED TRAIN GENERATOR
            train_generator = None

            # get data
            X_train = self.data.X_train
            y_train = self.data.y_train

            if self.StrategyList:  # if strategylist is not empty
                for i, j in zip(self.StrategyList, range(len(self.StrategyList))):
                    if isinstance(i, DataAugmentation.DataAugmentation):
                        train_generator = self.StrategyList[j].applyStrategy(self.data)
                    else:
                        X_train, y_train = self.StrategyList[j].applyStrategy(self.data)

            es_callback = EarlyStopping(monitor=config.VALIDATION_LOSS, patience=4)
            decrease_callback = ReduceLROnPlateau(monitor=config.LOSS,
                                                  patience=1,
                                                  factor=0.7,
                                                  mode='min',
                                                  verbose=1,
                                                  min_lr=0.000001)
            decrease_callback2 = ReduceLROnPlateau(monitor=config.VALIDATION_LOSS,
                                                   patience=1,
                                                   factor=0.7,
                                                   mode='min',
                                                   verbose=1,
                                                   min_lr=0.000001)

            if train_generator is None:  # NO DATA AUGMENTATION

                history = model.fit(
                    x=X_train,
                    y=y_train,
                    batch_size=args[0],
                    epochs=config.EPOCHS,
                    validation_data=(self.data.X_val, self.data.y_val),
                    shuffle=True,
                    callbacks=[es_callback, decrease_callback, decrease_callback2],
                    # class_weight=config.class_weights
                    verbose=config.TRAIN_VERBOSE
                )

                return history, model

            # ELSE APPLY DATA AUGMENTATION
            history = model.fit_generator(
                generator=train_generator,
                validation_data=(self.data.X_val, self.data.y_val),
                epochs=config.EPOCHS,
                steps_per_epoch=X_train.shape[0] / args[0],
                shuffle=True,
                # class_weight=config.class_weights,
                verbose=config.TRAIN_VERBOSE,
                callbacks=[es_callback, decrease_callback, decrease_callback2]
            )

            return history, model

        except:
            raise CustomError.ErrorCreationModel(config.ERROR_ON_TRAINING)

    def __str__(self):
        pass