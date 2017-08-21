#!/usr/bin/env python3

import os

path = os.path.abspath(__file__)
SRC_FOLDER = os.path.dirname(path)
MAIN_FOLDER = os.path.dirname(SRC_FOLDER)
DATA_FOLDER = os.path.join(MAIN_FOLDER, "data/")
MODELS_FOLDER = os.path.join(MAIN_FOLDER, "models/")
TEST_FOLDER = os.path.join(MAIN_FOLDER, "test/")
TMP_FOLDER = os.path.join(MAIN_FOLDER, "tmp/")
