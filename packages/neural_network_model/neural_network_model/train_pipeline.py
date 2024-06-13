import os
import logging
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from neural_network_model import pipeline as pipe
from neural_network_model.config import config
from neural_network_model.processing import data_management as dm
from neural_network_model.processing import preprocessors as pp

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_training(save_result: bool = True):
    """Train a Convolutional Neural Network."""

    _logger.info('Loading image paths...')
    images_df = dm.load_image_paths(config.DATA_FOLDER)
    _logger.info('Splitting train and test data...')
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)
    _logger.info('Encoding target...')
    enc = pp.TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)

    _logger.info('Fitting the pipeline...')
    pipe.pipe.fit(X_train, y_train)

    if save_result:
        _logger.info('Saving the encoder...')
        joblib.dump(enc, config.ENCODER_PATH)
        _logger.info('Saving the encoder...')
        dm.save_pipeline_keras(pipe.pipe)
    _logger.info('Training complete.')


if __name__ == '__main__':
    run_training(save_result=True)
