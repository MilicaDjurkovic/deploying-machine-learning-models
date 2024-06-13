# The Keras model loading function does not play well with
# Pathlib at the moment, so we are using the old os module
# style

import os
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
DATA_FOLDER = os.path.join(DATASET_DIR, 'v2-plant-seedlings-dataset')

# MODEL PERSISTING
MODEL_NAME = 'cnn_model'
PIPELINE_NAME = 'cnn_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'

# MODEL FITTING
IMAGE_SIZE = 150  # 50 for testing, 150 for final model
BATCH_SIZE = 5
EPOCHS = int(os.environ.get('EPOCHS', 1))  # 1 for testing, 10 for final model

_logger.info(f"Current working directory: {PWD}")
_logger.info(f"Package root: {PACKAGE_ROOT}")
_logger.info(f"Dataset directory: {DATASET_DIR}")
_logger.info(f"Trained model directory: {TRAINED_MODEL_DIR}")
_logger.info(f"Data folder: {DATA_FOLDER}")


with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.h5'
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

CLASSES_FILE_NAME = f'{CLASSES_NAME}_{_version}.pkl'
CLASSES_PATH = os.path.join(TRAINED_MODEL_DIR, CLASSES_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)


_logger.info(f"Model path: {MODEL_PATH}")
_logger.info(f"Pipeline path: {PIPELINE_PATH}")
_logger.info(f"Classes path: {CLASSES_PATH}")
_logger.info(f"Encoder path: {ENCODER_PATH}")