#!/usr/bin/env python3

import os
import time
import random
random.seed(time.time())

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
import pickle
import numpy
from skimage.feature import canny

from src import ImageSplitter

from src.Settings import CHANNELS
from src.Settings import PREPROCESSED_LAYERS
from src.Settings import IMAGE_WIDTH_IDENTIFY
from src.Settings import IMAGE_HEIGHT_IDENTIFY
from src.Settings import IMAGE_WIDTH_REMOVE
from src.Settings import IMAGE_HEIGHT_REMOVE
from src.Settings import BATCH_SIZE

class Generator():
    '''
        Base generator
        Interface:
            - flow
            - getBatch
    '''
    _source = None
    _source_masks = None
    _image_splitter = None
    _image_width = None
    _image_height = None
    _force_image_size = False
    _preprocessed_layers = 0

    def __init__(self, source=None, source_masks=None):
        self._source = source
        self._source_masks = source_masks
        self._image_splitter = ImageSplitter(CHANNELS)

    def setSource(self, source):
        self._source = source

    def getSource(self):
        return self._source

    def setSourceMasks(self, source_masks):
        self._source_masks = source_masks

    def getSourceMasks(self):
        return self._source_masks

    def getImageSplitter(self):
        return self._image_splitter

    def setImageSize(self, image_width, image_height):
        self._image_width = image_width
        self._image_height = image_height

    def getImageSize(self):
        return self._image_width, self._image_height

    def setPreprocessedLayers(self, layers):
        self._preprocessed_layers = layers

    def getPreprocessedLayers(self, layers):
        return self._preprocessed_layers

    def getForceImageSize(self):
        return self._force_image_size

    def setForceImageSize(self, value):
        self._force_image_size = value

    def flow(self):
        raise NotImplemented

    def getBatch(self):
        raise NotImplemented

class ImageGenerator(Generator):
    '''
        Image processing to feed a neural network.
        Implements:
            - getInputData
            - preprocessImage
    '''

    def preprocessImage(self, img):
        '''
            Apply some kind of pre-processing to an image.
            Input:
                - img: a PIL.Image object
            Returns:
                - processed_image: numpy array
        '''
        w, h = img.size
        processed_image = numpy.zeros(
                            (h, w, self._preprocessed_layers),
                            dtype="float32")
        # Compute edges
        img_g = img.convert('L')
        data = numpy.asarray(img_g, dtype="float32") / 255.
        edges = canny(data)
        processed_image[:,:,-1] = numpy.float32(edges)
        return processed_image


    def modifyImage(self, img):
        '''
            Modify an image.
            Input:
                - img: PIL.Image
            Returns:
                - result_image: modified PIL.Image
                - process: operation applied
        '''
        return img, None


    def getInputData(self, img):
        '''
            Get the data to feed to the network.
            Input:
                - img: PIL.Image
            Returns:
                - data: numpy array
                - process: process applied to the image
        '''
        # Resize the image if necessary
        w, h = img.size
        if self._force_image_size and (w != self._image_width or h != self._image_height):
            img = img.resize((self._image_width, self._image_height), Image.LINEAR)
        # Process the image
        img, process = self.modifyImage(img)
        base_data = self._image_splitter.getDataFromImage(img) / 255.
        # Pre-process the image
        processed_data = self.preprocessImage(img)
        # Put all together
        h, w, c = base_data.shape
        data = numpy.zeros(
                    (h, w, CHANNELS+self._preprocessed_layers),
                    dtype="float32")
        data[:, :, 0:CHANNELS] = base_data
        if self._preprocessed_layers > 0:
            data[:, :, CHANNELS:CHANNELS+self._preprocessed_layers] = processed_data
        return data, process


class WrongSourceType(BaseException):
    '''
        Wrong source type exception
    '''
    pass



def standardizer(x):
    '''
        Replace low numbers with 0.0 and others with 1.0
    '''
    if x < 0.1:
        return 0
    else:
        return 1

standardizer_numpy = numpy.vectorize(standardizer)


class BaseGenerator(ImageGenerator):
    '''
        Generate training examples
    '''

    def _pickColor(self, solid=True):
        '''
            Pick a color
        '''
        if solid:
            # returns white 1/2 of the times
            if random.random() < 0.5:
                return 255, 255, 255, 255
            else:
                self._pickColor(solid=False)
        else:
            # returns a random color
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            a = 255
            return r, g, b, a

    def _addSolidBackground(self, image):
        '''
            Add a solid background
        '''
        background = Image.new("RGBA", image.size, color=self._pickColor(solid=True))
        background.paste(image, (0,0), image)
        background = background.convert("RGB")
        return background


    def _addGradientBackground(self, image):
        '''
            Add a solid background
            TODO
        '''
        raise NotImplemented


    def addBackground(self, image):
        '''
            Add a solid color background or a gradient
        '''
        image = self._addSolidBackground(image)
        '''
        if random.random() < 0.5:
            image = self._addSolidBackground(image)
        else:
            image = self._addGradientBackground(image)
        '''
        if CHANNELS == 3:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        return image


class IdentifyGenerator(BaseGenerator):
    _force_image_size = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPreprocessedLayers(PREPROCESSED_LAYERS)
        self.setImageSize(IMAGE_WIDTH_IDENTIFY, IMAGE_HEIGHT_IDENTIFY)

    def modifyImage(self, image):
        '''
            Add a random background and reshape the mask
        '''
        mask = image.split()[-1]
        image = self.addBackground(image)
        image = image.convert("RGB")
        # Convert mask to numpy array and reshape
        mask = numpy.asarray(mask) / 255.
        mask = mask.reshape(self._image_width*self._image_height)
        return image, mask


    def flow(self, batch_size=BATCH_SIZE, finite=False):
        '''
            Iterate over all the training set
        '''
        self._files = os.listdir(self._source)
        x = []
        labels = []
        img_index = -1
        while 1:
            # Load a new image
            img_index += img_index
            if finite and img_index == len(self._files):
                if len(x) == 0:
                    return None
                else:
                    x = numpy.asarray(x, dtype='float32')
                    labels = numpy.asarray(labels, dtype='float32')
                    return (x, labels)
            img_index = img_index % len(self._files)
            fname = self._files[img_index]
            rpath = os.path.join(self._source, fname)
            img = Image.open(rpath).convert("RGBA")
            img_data, label = self.getInputData(img)
            x.append(img_data)
            labels.append(label)
            if len(labels) == batch_size:
                x = numpy.asarray(x, dtype='float32')
                labels = numpy.asarray(labels, dtype='float32')
                yield (x, labels)
                x = []
                labels = []

    def getBatch(self, paths):
        keep = True
        x = []
        for path in paths:
            img = Image.open(path)
            iw, ih = img.size
            if iw != self._image_width or ih != self._image_height:
                img = img.resize((self._image_width, self._image_height), resample=Image.BICUBIC)
            img = img.convert("RGBA")
            img_data, _ = self.getInputData(img)
            x.append(img_data)
        return numpy.asarray(x, dtype='float32')

class LocalGenerator(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPreprocessedLayers(1)
        self.setImageSize(IMAGE_WIDTH_IDENTIFY, IMAGE_HEIGHT_IDENTIFY)

    def preprocessImage(self, img):
        return numpy.zeros((self._image_height, self._image_width, self._preprocessed_layers),
                            dtype="float32")

    def modifyImage(self, image):
        '''
            Add a random background and reshape the mask
        '''
        mask = image.split()[-1]
        image = self.addBackground(image)
        image = image.convert("RGB")
        # Convert mask to numpy array and reshape
        mask = numpy.asarray(mask) / 255.
        mask = mask.flatten()
        return image, mask

    def getInputData(self, img, mask):
        data, label = super().getInputData(img)
        data[:, :, -1] = mask
        return data, label

    def splitImageAndMask(self, img, mask, stride=None):
        # Setup stride
        if stride is None:
            stride = (self._image_width, self._image_height)
        stride_w, stride_h = stride
        # Split image
        img_splits = []
        for split in self._image_splitter.splitImage(img, (self._image_width, self._image_height), padding=False, stride=stride):
            img_splits.append(split)
        # Split masks_local
        mask_splits = []
        h, w = mask.shape
        a = numpy.floor((w - self._image_width) / stride_w) + 1
        b = numpy.floor((h - self._image_height) / stride_h) + 1
        for i in range(int(a)):
            for j in range(int(b)):
                x = i * stride_w
                y = j * stride_h
                mask_splits.append(mask[y : y + self._image_height, x : x + self._image_height])
        return img_splits, mask_splits

    def _getMaskData(self, path):
        with open(path, 'rb') as hand:
            data = hand.read()
        return pickle.loads(data)

    def flow(self, batch_size=BATCH_SIZE, finite=False):
        '''
            Iterate over all the training set
        '''
        self._files = os.listdir(self._folder)
        x = []
        labels = []
        img_index = -1
        while 1:
            # Load a new image
            img_index += img_index
            if finite and img_index == len(self._files):
                if len(x) == 0:
                    return None
                else:
                    x = numpy.asarray(x, dtype='float32')
                    labels = numpy.asarray(labels, dtype='float32')
                    return (x, labels)
            img_index = img_index % len(self._files)
            fname = self._files[img_index]
            rpath = os.path.join(self._source, fname)
            mpath = os.path.join(self._source_masks, fname + '.mask')
            img = Image.open(rpath).convert("RGBA")
            mask = self._getMaskData(mpath)
            img_splits, mask_splits = self.splitImageAndMask(img, mask)
            for index in range(len(img_splits)):
                img_split = img_splits[index]
                mask_split = mask_splits[index]
                img_data, label = self.getInputData(img_split, mask_split)
                x.append(img_data)
                labels.append(label)
            if len(labels) >= batch_size:
                x = numpy.asarray(x, dtype='float32')
                labels = numpy.asarray(labels, dtype='float32')
                yield (x, labels)
                x = []
                labels = []

    def getBatch(self, paths):
        keep = True
        x = []
        for ipath, mpath in paths:
            img = Image.open(ipath).convert('RGBA')
            mask = self._getMaskData(mpath)
            iw, ih = img.size
            img_splits, mask_splits = self.splitImageAndMask(img, mask)
            for index in range(len(img_splits)):
                img_split = img_splits[index]
                mask_split = mask_splits[index]
                img_data, _ = self.getInputData(img_split, mask_split)
                x.append(img_data)
        return numpy.asarray(x, dtype='float32')

class RemoveGenerator(BaseGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPreprocessedLayers(1)
        self.setImageSize(IMAGE_WIDTH_REMOVE, IMAGE_HEIGHT_REMOVE)

    def _getMaskData(self, path):
        with open(path, 'rb') as hand:
            data = hand.read()
        return pickle.loads(data)

    def modifyImage(self, image):
        '''
            Add a random background and reshape the mask
        '''
        mask = image.split()[-1]
        image = self.addBackground(image)
        image = image.convert("RGB")
        # Convert mask to numpy array and reshape
        mask = numpy.asarray(mask) / 255.
        mask = mask.flatten()
        return image, mask

    def resizeMask(self, mask, size):
        return self._image_splitter.upscaleMatrix(mask, size)

    def getInputData(self, img, mask):
        data, label = super().getInputData(img)
        data[:, :, -1] = mask
        return data, label

    def flow(self, batch_size=3):
        '''
            Iterate over all the training set
            :param batch_size
            :yields x, labels
        '''
        self._files = os.listdir(self._folder)
        x = []
        labels = []
        img_index = -1
        while 1:
            # Load a new image
            img_index += 1
            img_index = img_index % len(self._files)
            fname = self._files[img_index]
            path = os.path.join(self._source, fname)
            mpath = os.path.join(self._source_masks, fname + '.mask')
            img = Image.open(path).convert("RGBA")
            mask = self._getMaskData(mpath)
            img_data, label = self.getInputData(img, mask)
            x.append(img_data)
            labels.append(label)
            if len(labels) >= batch_size:
                x = numpy.asarray(x, dtype='float32')
                labels = numpy.asarray(labels, dtype='float32')
                yield (x, labels)
                x = []
                labels = []

    def getBatch(self, paths):
        keep = True
        x = []
        for ipath, mpath in paths:
            img = Image.open(ipath).convert("RGBA")
            mask = self._getMaskData(mpath)
            img_data, _ = self.getInputData(img, mask)
            x.append(img_data)
        return numpy.asarray(x, dtype='float32')
