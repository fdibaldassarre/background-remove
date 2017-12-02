#!/usr/bin/env python3

import os
import shutil
from PIL import Image

from src.Places import DATA_FOLDER
from src.Settings import IMAGE_WIDTH_IDENTIFY
from src.Settings import IMAGE_HEIGHT_IDENTIFY
from src.Settings import IMAGE_WIDTH_REMOVE
from src.Settings import IMAGE_HEIGHT_REMOVE

def resizeIn(base_folder, target_folder, size):
    # Prepare the target folder
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    # Iterate over the samples
    for filename in os.listdir(base_folder):
        if filename.startswith("."):
    	    continue
        path = os.path.join(base_folder, filename)
        target = os.path.join(target_folder, filename)
        image = Image.open(path)
        image = image.resize(size, resample=Image.LINEAR)
        image.save(target)

sizes = {
    "identify" : (IMAGE_WIDTH_IDENTIFY, IMAGE_HEIGHT_IDENTIFY),
    "local": (2*IMAGE_WIDTH_IDENTIFY, 2*IMAGE_HEIGHT_IDENTIFY),
    "remove" : (IMAGE_WIDTH_REMOVE, IMAGE_HEIGHT_REMOVE)
}

# Resize all the train images
train_folder = os.path.join(DATA_FOLDER, "train")
print("Training set")
for target in sizes:
    size = sizes[target]
    print(target, "--", size)
    target_folder = os.path.join(DATA_FOLDER, "auto-generated/train/resized_" + target)
    resizeIn(train_folder, target_folder, size)

# Resize all the validation images
validation_folder = os.path.join(DATA_FOLDER, "validation")
print("Validation set")
for target in sizes:
    size = sizes[target]
    print(target, "--", size)
    target_folder = os.path.join(DATA_FOLDER, "auto-generated/validation/resized_" + target)
    resizeIn(validation_folder, target_folder, size)
