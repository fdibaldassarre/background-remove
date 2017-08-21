#!/usr/bin/env python3

import os
import shutil
import pickle
import numpy
import scipy
from PIL import Image

from src.Models import IdentifyModel
from src.Generators import IdentifyGenerator
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Settings import IMAGE_WIDTH_IDENTIFY as IMAGE_WIDTH
from src.Settings import IMAGE_HEIGHT_IDENTIFY as IMAGE_HEIGHT
from src.Utils import upscaleMatrix


print("Load model")
model_path = os.path.join(MODELS_FOLDER, "identify.h5")
model = IdentifyModel()
model.load(model_path)

batch_size = 100

'''
def upscaleMatrix(kernelIn):
    kernelIn = kernelIn.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))
    kernelOut = numpy.zeros((2 * IMAGE_HEIGHT, 2 * IMAGE_WIDTH), dtype='float32')
    h, w = kernelOut.shape
    for y in range(h):
        for x in range(w):
            kernelOut[y, x] = kernelIn[int(y/2), int(x/2)]
    return kernelOut
'''

def saveBatch(generator, fnames, folder_name, target_folder):
    # Prepare batch by taking files in the resized folder
    resized_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/resized_identify')
    paths = []
    for fname in fnames:
        paths.append(os.path.join(resized_folder, fname))
    # Predict on batch
    batch = generator.getBatch(path)
    result = model.predict_on_batch(batch)
    N, _ = result.shape
    result = result.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    # Save the batch
    for index in range(N):
        # Setup savepaths
        fname = fnames[index]
        mask_path = os.path.join(target_folder, fname + '.mask')
        # Upscale the mask
        img_data = result[index, :, :, 0]
        image_up = upscaleMatrix(img_data, (2*IMAGE_HEIGHT, 2*IMAGE_WIDTH))
        # Save the mask
        data = pickle.dumps(image_up, protocol=0)
        with open(mask_path, 'wb') as hand:
            hand.write(data)

def computeAndResize(folder_name):
    base_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/resized_identify')
    # Create masks folder
    target_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/masks_local')
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    # Initialize generator
    generator = IdentifyGenerator()
    # Iterate over the images list
    fnames = []
    for fname in os.listdir(base_folder):
        fnames.append(fname)
        if len(fnames) >= batch_size:
            saveBatch(generator, fnames, folder_name, target_folder)
            # Reset fnames
            fnames = []
    if len(fnames) > 0:
        saveBatch(generator, fnames, folder_name, target_folder)
        # Reset fnames
        fnames = []

# Resize all the train images
computeAndResize("train")

# Resize all the validation images
computeAndResize("validation")
