#!/usr/bin/env python3

import sys
import os
import argparse

path = "./image.png"
output_path = "./image_new.png"

parser = argparse.ArgumentParser(description="BlackBox predict")
parser.add_argument('input_image',default=None,
                    help='Input image')
parser.add_argument('output_image', default=None,
                    help='Output image')

args = parser.parse_args()

input_path = args.input_image
output_path = args.output_image

if input_path is None or output_path is None:
    print("Missing arguments")
    sys.exit(1)

if not os.path.exists(input_path):
    print("Input image does not exist")
    sys.exit(2)

print("Input: " + input_path)

from PIL import Image
import numpy

from src.Generators import IdentifyGenerator
from src.Generators import LocalGenerator
from src.Generators import RemoveGenerator
from src.Models import IdentifyModel
from src.Models import LocalModel
from src.Models import RemoveModel
from src.Places import MODELS_FOLDER
from src.Utils import upscaleMatrix

## Open image
img = Image.open(input_path).convert("RGBA")
img_size = img.size

## 1 - Identify Image

print("Identifing image")
# Load the model
model_path = os.path.join(MODELS_FOLDER, "identify.h5")
model = IdentifyModel()
model.load(model_path)
# Load the generator
generator = IdentifyGenerator()
# Prepare the batch
data, _ = generator.getInputData(img)
h, w, c = data.shape
batch = data.reshape((1, h, w, c))
# Predict
mask = model.predict_on_batch(batch)[0, :]
mask = mask.reshape((h, w))

# Check if I can apply the next model
model_path = os.path.join(MODELS_FOLDER, "local.h5")
if not os.path.exists(model_path):
    print("Identify result")
    mask = mask.reshape((h, w, 1))
    img = generator.getImageSplitter().getImageFromData(mask)
    img.save(output_path)
    # Save mask and exit
    sys.exit(0)
## 2 - Refine local prediction
print("Refine local prediction")
# Load the model
model = LocalModel()
model.load(model_path)
# Load the generator
generator = LocalGenerator()
# Image preprocesing: 2x upscaling
from src.Settings import IMAGE_WIDTH_IDENTIFY
from src.Settings import IMAGE_HEIGHT_IDENTIFY
up_size = (2*IMAGE_WIDTH_IDENTIFY, 2*IMAGE_HEIGHT_IDENTIFY)
stride = (int(IMAGE_WIDTH_IDENTIFY/4), int(IMAGE_HEIGHT_IDENTIFY/4))
# Resize the image
img2 = img.resize(up_size, Image.LINEAR)
# Upscale the mask
w, h = up_size
mask = upscaleMatrix(mask, (h, w))
# Prepare batch
batch = []
img_splits, mask_splits = generator.splitImageAndMask(img2, mask, stride=stride)
for index in range(len(img_splits)):
    img_split = img_splits[index]
    mask_split = mask_splits[index]
    img_data, _ = generator.getInputData(img_split, mask_split)
    batch.append(img_data)
batch = numpy.asarray(batch, dtype="float32")
# Predict
masks = model.predict_on_batch(batch)
masks = masks.reshape((-1, IMAGE_HEIGHT_IDENTIFY, IMAGE_WIDTH_IDENTIFY))
# Recompose mask
mask = generator.getImageSplitter().recomposeImage(masks, up_size, stride=stride)

# Check if I can apply the next model
model_path = os.path.join(MODELS_FOLDER, "remove.h5")
if not os.path.exists(model_path):
    print("Local result")
    img = generator.getImageSplitter().getImageFromData(mask)
    img.save(output_path)
    # Save mask and exit
    sys.exit(0)

# Reshape mask
h, w, _ = mask.shape
mask = mask.reshape((h, w))

## 3 - Remove background
print("Remove background")
# Load the model
w, h = img_size
model = RemoveModel(shape=(h, w))
model.load(model_path)
# Load the generator
generator = RemoveGenerator()
generator.setForceImageSize(False)
# Upscale mask
mask = upscaleMatrix(mask, (h, w))
# Get input data and create batch
data, _ = generator.getInputData(img, mask)
batch = data.reshape((1, *data.shape))
# Predict
result = model.predict_on_batch(batch)[0, :]
# Reshape
result = result.reshape((h, w, 1))
# Get alpha image
alpha = generator.getImageSplitter().getImageFromData(result)
# Compose
img.putalpha(alpha)
# Save
print("Save final result")
img.save(output_path)
