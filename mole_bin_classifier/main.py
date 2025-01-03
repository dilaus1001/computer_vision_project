import logging
import os
from pathlib import Path
import tensorflow as tf
import warnings

from model import create_mobile_model
from train import train_model, evaluate_model
from preprocess import prepare_datasets

class Config:
    INPUT_DIR = '/Users/francesco/Repository/computer_vision_project/dataset/mix_dataset/categorized_images'
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = '/Users/francesco/Repository/computer_vision_project/saved_models'
    BALANCED_DIR = '/Users/francesco/Repository/computer_vision_project/dataset/mix_dataset/balanced_images'
    
    BATCH_SIZE = 128
    EPOCHS = 1
    LEARNING_RATE = 0.001
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            DEVICE = '/GPU:0'
        except RuntimeError as e:
            logging.warning(f'Error configuring GPUs: {e}')
            DEVICE = '/CPU:0'
    else:
        DEVICE = '/CPU:0'

def setup_logging(output_dir: Path):
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main Pipeline:
    - Setting up directories and logging
    - Preparing datasets
    - Creating and training the model
    - Evaluating performance
    - Saving the trained model
    """
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir)
    logging.info(f'Starting training with configuration: {vars(Config)}')
    logging.info(f'Using device: {Config.DEVICE}')
    
    try:
        with tf.device(Config.DEVICE):
            logging.info('Preparing datasets...')
            train_dataset, val_dataset, test_dataset = prepare_datasets(
                data_dir=Config.BALANCED_DIR,
                batch_size=Config.BATCH_SIZE
            )
            
            model = create_mobile_model(num_classes=1)
            
            logging.info('Starting training...')
            history = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                checkpoint_dir=output_dir / 'checkpoints',
                epochs=Config.EPOCHS,
                initial_learning_rate=Config.LEARNING_RATE
            )
            
            logging.info('Evaluating model...')
            evaluate_model(model, test_dataset)
            
            logging.info('Saving final model...')
            model.save(Config.MODEL_DIR / 'skc_model.h5')
            
            logging.info('Training completed successfully!')
            
    except Exception as e:
        logging.error(f'Error during execution: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    
    main()