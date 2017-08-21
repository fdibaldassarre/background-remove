#!/usr/bin/env python3

import os

from src.Models import IdentifyModel
from src.Places import MODELS_FOLDER
from src.Places import DATA_FOLDER
from src.Generator import TrainIdentifyGenerator

print("Initialize model")
model = IdentifyModel()
model_path = os.path.join(MODELS_FOLDER, "identify.h5")
# Check if old models exist
if os.path.exists(model_path):
    print("Load model")
    model.load(model_path)
# Train
print("Train model")
train_data = os.path.join(DATA_FOLDER, "train/resized")
validation_data = os.path.join(DATA_FOLDER, "validation/resized")
generator_train = TrainIdentifyGenerator(train_data)
generator_validation = TrainIdentifyGenerator(validation_data)
model.fit_generator(generator_train, generator_validation)
print("Save")
model.save(model_path)
