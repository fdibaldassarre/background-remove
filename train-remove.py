#!/usr/bin/env python3

import os

from src.Models import RemoveModelTrain
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Generator import RemoveGenerator

print("Initialize model")
model = RemoveModelTrain()
model_path = os.path.join(MODELS_FOLDER, "remove.h5")
# Check if old models exist
if os.path.exists(model_path):
    print("Load model")
    model.load(model_path)
# Generator train
train_data = os.path.join(DATA_FOLDER, "auto-generated/train/resized_remove")
train_masks = os.path.join(DATA_FOLDER, "auto-generated/train/masks_remove")
generator_train = RemoveGenerator(train_data, train_masks)
# Generator validation
validation_data = os.path.join(DATA_FOLDER, "auto-generated/validation/resized_remove")
validation_masks = os.path.join(DATA_FOLDER, "auto-generated/validation/masks_remove")
generator_validation = RemoveGenerator(validation_data, validation_masks)
# Fit model
print("Train model")
model.fit_generator(generator_train, generator_validation)
# Save
print("Save")
model.save(model_path)
