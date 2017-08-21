# background-remove
Remove simple backgrounds from images (work in progress).

## Requirements

- Python 3
- numpy
- PIL
- scikit-image
- Keras

## Training

Put the training and validation images in the folders data/train and data/validation.
In these images the background should be transparent.

The training phase is divided in 3 phases.

### Training identify

Run
```sh
./prepareDataIdentify.py
```
and then
```sh
./train-identify.py
```

### Training local

Run
```sh
./prepareDataLocal.py
```
and then
```sh
./train-local.py
```

### Training remove

Run
```sh
./prepareDataRemove.py
```
and then
```sh
./train-remove.py
```

## Usage

To remove the background from an image run

```sh
./predict.py input_path output_path
```

## TODO
- Improve the models and the parameters used to train.
- Better remove training
- Proper documentation for each class
