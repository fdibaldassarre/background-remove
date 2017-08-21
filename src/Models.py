#!/usr/bin/env python3

from keras import initializers

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Concatenate

from keras.models import Model
from keras.optimizers import Adam

import numpy

from .Settings import IMAGE_WIDTH_IDENTIFY
from .Settings import IMAGE_HEIGHT_IDENTIFY
from .Settings import IMAGE_WIDTH_REMOVE
from .Settings import IMAGE_HEIGHT_REMOVE
from .Settings import CATEGORIES
from .Settings import CHANNELS
from .Settings import PREPROCESSED_LAYERS
from .Settings import BATCH_SIZE
from .Settings import BATCH_SIZE_VALIDATION
from .Settings import EPOCHS
from .Settings import PRETRAIN_EPOCHS
from .Settings import STEPS_PER_EPOCH

class Base():
    _epochs = EPOCHS
    _batch_size = BATCH_SIZE
    _batch_size_validation = BATCH_SIZE_VALIDATION

    def __init__(self):
        self.loadModel()

    def loadModel(self):
        raise NotImplemented

    def setEpochs(self, epochs):
        self._epochs = epochs

    def getEpochs(self):
        return self._epochs

    def setBatchSize(self, batch_size):
        self._batch_size = batch_size

    def getBatchSize(self):
        return self._batch_size

    def setBatchSizeValidation(self, batch_size):
        self._batch_size_validation = batch_size

    def getBatchSizeValidation(self):
        return self._batch_size_validation

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)

    def fit_generator(self, generator_train, generator_validation):
        self.model.fit_generator(
                        generator_train.flow(batch_size=self._batch_size),
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_data=generator_validation.flow(batch_size=self._batch_size_validation),
                        validation_steps=1,
                        epochs=self._epochs,
                        verbose=1)

    def predict_on_batch(self, batch):
        return self.model.predict_on_batch(batch)


class IdentifyModel(Base):
    '''
        U-net model
    '''
    def loadModel(self):
        inputs = Input(shape=(64, 64, CHANNELS + PREPROCESSED_LAYERS, ))
        down_0 = inputs # 64x64
        down_1, rem_1 = self._down(down_0, 32) # 32x32
        down_2, rem_2 = self._down(down_1, 64) # 16x16
        down_3, rem_3 = self._down(down_2, 128) # 8x8
        #down_4, rem_4 = self._down(down_3, 256) # 4x4
        up_0 = down_3 # 4x4
        #dense_0 = Flatten()(down_4) # 4*4*256 = 4096
        #dense_1 = Dense(2048, activation='relu')(dense_0)
        #dense_1b = Dropout(0.5)(dense_1)
        #dense_2 = Dense(4096, activation='relu')(dense_1b)
        #up_0 = Reshape(target_shape=(4, 4, 256))(dense_2) # 4x4
        up_1 = self._up(up_0, rem_3, 64) # 8x8
        up_2 = self._up(up_1, rem_2, 32) # 16x16
        up_3 = self._up(up_2, rem_1, 1) # 32x32
        #up_4 = self._up(up_3, rem_1, 1, activator='softmax') # 64x64
        prediction = Flatten()(up_3)
        self.model = Model(inputs, prediction)
        self.model.compile(
                optimizer=Adam(1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    def _weightInitializer(self, stddev):
        return initializers.RandomNormal(mean=0.0, stddev=stddev, seed=None)

    def _biasInitializer(self):
        return initializers.Constant(value=0.1)

    def _down(self, layer, filters):
        input_size = int(layer.get_shape()[-1])
        stddev = numpy.sqrt(2.0 / (3*3*input_size))
        conv1 = Conv2D( filters,
                        kernel_size=(3,3),
                        padding='same',
                        activation='relu',
                        kernel_initializer=self._weightInitializer(stddev),
                        bias_initializer=self._biasInitializer()
                        )(layer)
        conv2 = Conv2D( filters,
                        kernel_size=(3,3),
                        padding='same',
                        activation='relu',
                        kernel_initializer=self._weightInitializer(stddev),
                        bias_initializer=self._biasInitializer()
                        )(conv1)
        max_pool = MaxPool2D(pool_size=(2,2))(conv2)
        return max_pool, conv2

    def _up(self, layer, conv, filters, activator=None):
        if activator is None:
            activator = 'relu'
        stddev = numpy.sqrt(2.0 / (3*3*filters))
        upconv = Conv2DTranspose(   filters,
                                    kernel_size=(3,3),
                                    strides=(2,2),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=self._weightInitializer(stddev),
                                    bias_initializer=self._biasInitializer()
                                    )(layer)
        concat = Concatenate(axis=3)([conv, upconv])
        input_size = int(concat.get_shape()[-1])
        stddev = numpy.sqrt(2.0 / (3*3*input_size))
        conv1 = Conv2D( filters,
                        kernel_size=(3,3),
                        padding='same',
                        activation=activator,
                        kernel_initializer=self._weightInitializer(stddev),
                        bias_initializer=self._biasInitializer()
                        )(concat)
        return conv1

class LocalModel(Base):

    def __init__(self):
        super().__init__()
        self.setEpochs(5)
        self.setBatchSize(400)

    def loadModel(self):
        # Load model
        inputs = Input(shape=(IMAGE_HEIGHT_IDENTIFY, IMAGE_WIDTH_IDENTIFY, CHANNELS + 1, ))
        conv_0 = inputs
        conv_1 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(conv_0)
        conv_2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(conv_1)
        conv_2b = Dropout(0.5)(conv_2)
        conv_3 = Conv2D(1, kernel_size=(3,3), activation='relu', padding='same')(conv_2b)
        prediction = Flatten()(conv_3)
        self.model = Model(inputs, prediction)
        self.model.compile(
                optimizer=Adam(lr=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy'])


class RemoveModelTrain(Base):

    def __init__(self):
        super().__init__()
        self.setEpochs(10)
        self.setBatchSize(1)
        self.setBatchSizeValidation(1)

    def getInputShape(self):
        return IMAGE_HEIGHT_REMOVE, IMAGE_WIDTH_REMOVE

    def loadModel(self):
        height, width = self.getInputShape()
        inputs = Input(shape=(height, width, CHANNELS + 1, ))
        # 3-Conv
        conv_0 = inputs
        conv_1 = Conv2D(16, kernel_size=(3,3), activation='relu', padding='same')(conv_0)
        conv_2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(conv_1)
        conv_3 = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(conv_2)
        conv_4 = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(conv_3)
        conv_5 = Conv2D(1, kernel_size=(3,3), activation='relu', padding='same')(conv_4)
        prediction = Flatten()(conv_5)
        self.model = Model(inputs, prediction)
        self.model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])

class RemoveModel(RemoveModelTrain):
    _input_shape = None
    
    def __init__(self, shape):
        self.setInputShape(shape)
        super().__init__()

    def getInputShape(self):
        return self._input_shape

    def setInputShape(self, shape):
        self._input_shape = shape

def getIdentifyModel(*args, **kwargs):
    return IdentifyModel(*args, **kwargs)

def getIdentifyLocalModel(*args, **kwargs):
    return IdentifyLocalModel(*args, **kwargs)

def getRemoveModel(*args, **kwargs):
    return RemoveModel(*args, **kwargs)
