#!/usr/bin/env python3

import os

from src.Models import IdentifyLocalModel
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Generator import LocalGenerator

print("Initialize model")
model = IdentifyLocalModel()
model_path = os.path.join(MODELS_FOLDER, "local.h5")
# Check if old models exist
if os.path.exists(model_path):
    print("Load model")
    model.load(model_path)
# Train generator
train_data = os.path.join(DATA_FOLDER, "auto-generated/train/resized_local")
train_masks = os.path.join(DATA_FOLDER, "auto-generated/train/masks_local")
generator_train = LocalGenerator(train_data, train_masks)
# Validation generator
validation_data = os.path.join(DATA_FOLDER, "auto-generated/validation/resized_local")
validation_masks = os.path.join(DATA_FOLDER, "auto-generated/validation/masks_local")
generator_validation = LocalGenerator(validation_data, validation_masks)
# Fit model
print("Train model")
model.fit_generator(generator_train, generator_validation)
# Save
print("Save")
model.save(model_path)
