from . import Model
import Data
from exceptions import CustomError
import config
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization, Input
from keras.callbacks.callbacks import History
from typing import Tuple
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from .Strategies_Train import Strategy, DataAugmentation
from keras import regularizers
from keras.models import Model as mp
from keras.initializers import glorot_uniform

class VGGNet(Model.Model):

    def __init__(self, data : Data.Data, *args):
        super(VGGNet, self).__init__(data, *args)

    def addStrategy(self, strategy : Strategy.Strategy) -> bool:
        return super(VGGNet, self).addStrategy(strategy)

    def add_stack(self, input, numberFilters, dropoutRate, input_shape=None):

        '''
        This function represents the implementation of stack cnn layers (in this case using only 2 cnn layers compacted)
        Conv --> Activation --> Conv --> Activation --> MaxPooling --> BatchNormalization --> Dropout
        :param input: tensor with current model architecture
        :param numberFilters: integer: number of filters to put on Conv layer
        :param dropoutRate: float (between 0.0 and 1.0)
        :param input_shape: tuple (height, width, channels) with shape of first cnn layer --> default None (not initial layers)
        :return: tensor of updated model
        '''

        try:

            if input_shape!=None:
                input = Conv2D(filters=numberFilters, kernel_size=(3, 3), strides=1, input_shape=input_shape,
                       padding=config.SAME_PADDING, kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            else:
                input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING,
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = Conv2D(filters=numberFilters, kernel_size=(3,3), strides=1, padding=config.SAME_PADDING,
                           kernel_regularizer=regularizers.l2(config.DECAY)) (input)
            input = Activation(config.RELU_FUNCTION) (input)
            input = BatchNormalization() (input)
            input = MaxPooling2D(pool_size=(2,2), strides=2) (input)
            input = Dropout(dropoutRate) (input)

            return input

        except:
            raise

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

            # definition of input shape and Input Layer
            input_shape = (config.WIDTH, config.HEIGHT, config.CHANNELS)
            input = Input(input_shape)

            # first stack convolution layer
            model = self.add_stack(input, args[1], 0.25, input_shape)

            ## add stack conv layer to the model
            numberFilters = args[1] + args[2]
            for i in range(args[0]):
                model = self.add_stack(model, numberFilters, 0.25)
                numberFilters += args[2]

            # flatten
            model = Flatten()(model)

            # Full Connected Layer(s)
            for i in range(args[3]):
                model = Dense(units=args[4], kernel_regularizer=regularizers.l2(config.DECAY))(model)
                model = Activation(config.RELU_FUNCTION)(model)
                model = BatchNormalization()(model)
                if i != (args[3] - 1):
                    model = Dropout(0.25) (model) ## applies Dropout on all FCL's except FCL preceding the ouput layer (softmax)

            # Output Layer
            model = Dense(units=config.NUMBER_CLASSES)(model)
            model = Activation(config.SOFTMAX_FUNCTION)(model)

            # build model
            model = mp(inputs=input, outputs=model)

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

            es_callback = EarlyStopping(monitor=config.VALIDATION_LOSS, patience=3)
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