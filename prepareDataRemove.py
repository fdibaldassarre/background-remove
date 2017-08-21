#!/usr/bin/env python3

import os
import shutil
import pickle
import numpy
import scipy
from PIL import Image

from src.Models import LocalModel
from src.Generators import LocalGenerator
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Settings import IMAGE_WIDTH_REMOVE as IMAGE_WIDTH
from src.Settings import IMAGE_HEIGHT_REMOVE as IMAGE_HEIGHT
from src.Utils import upscaleMatrix


print("Load model")
model_path = os.path.join(MODELS_FOLDER, "local.h5")
model = LocalModel()
model.load(model_path)

batch_size = 50

def saveBatch(generator, fnames, folder_name, target_folder):
    # Prepare batch by taking files in the resized folder
    resized_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/resized_local/')
    masks_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/masks_local/')
    paths = []
    for fname in fnames:
        rpath = os.path.join(resized_folder, fname)
        mpath = os.path.join(masks_folder, fname + '.mask')
        paths.append((rpath, mpath))
    # Predict on batch
    batch = generator.getBatch(paths)
    result = model.predict_on_batch(batch)
    # an image is the composition of 4 elements in the batch
    imgs = []
    w, h = generator.getImageSize()
    N, _ = result.shape
    for i in range(int(N/4)):
        img_data = result[4*i : 4*i + 4, :].reshape((-1, h, w))
        rec = generator.getImageSplitter().recomposeImage(img_data, (2*w, 2*h))
        imgs.append(rec)
    imgs = numpy.asarray(imgs)
    N = imgs.shape[0]
    # Save the batch
    for index in range(N):
        fname = fnames[index]
        savepath = os.path.join(target_folder, fname)
        mask_path = savepath + '.mask'
        img_data = imgs[index, :, :, 0]
        mask_data = upscaleMatrix(img_data, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Save the mask
        data = pickle.dumps(mask_data, protocol=0)
        with open(mask_path, 'wb') as hand:
            hand.write(data)

def computeMasks(folder_name):
    base_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/resized_remove')
    # Setup the folders
    target_folder = os.path.join(DATA_FOLDER, 'auto-generated/' + folder_name + '/masks_remove')
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    # Initialize the generator
    generator = LocalGenerator()
    # Iterate over the file list
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
computeMasks("train")

# Resize all the validation images
computeMasks("validation")
