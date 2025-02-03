import logging
from pathlib import Path
import tensorflow as tf
import warnings
import sys

from model import create_mobile_model
from train import train_model, evaluate_model
from preprocess import prepare_datasets

class Config:
    INPUT_DIR = Path('/Users/francesco/repos/computer_vision_project/src/integration_pipeline/src')
    OUTPUT_DIR = Path('outputs')
    MODEL_DIR = Path('/Users/francesco/repos/computer_vision_project/src/mole_bin_classifier/model_saved')

    # METADATA_PATH = INPUT_DIR / 'mix_dataset.csv'
    METADATA_PATH = '/Users/francesco/repos/computer_vision_project/dataset/mix_dataset/mix_dataset.csv'
    IMAGES_DIR = INPUT_DIR / 'cropped_detections'
    IMBALANCE_FACTOR = 1.1 

    BATCH_SIZE = 64
    EPOCHS = 1000
    LEARNING_RATE = 0.001
    
    gpus = tf.config.list_physical_devices('GPU') 
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            DEVICE = '/GPU:0'
        except RuntimeError as e:
            logging.warning(f'Error configuring GPU: {e}. Falling back to CPU.')
            DEVICE = '/CPU:0'
    else:
        mps_devices = tf.config.list_physical_devices('GPU')
        if mps_devices: 
            DEVICE = '/GPU:0'
        else:
            logging.warning('No GPU or MPS devices found. Falling back to CPU.')
            DEVICE = '/CPU:0'

def setup_logging(output_dir: Path):
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'training.log'
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

def save_models(model, h5_path, tflite_path):
    h5_path = Path(h5_path)
    tflite_path = Path(tflite_path)

    logging.info(f'Saving H5 model to: {h5_path}')
    try:
        model.save(str(h5_path))

        if h5_path.exists():
            size = h5_path.stat().st_size
            logging.info(f'H5 model saved successfully. File size: {size/1024/1024:.2f} MB')
        else:
            raise FileNotFoundError(f'Model file not found after saving: {h5_path}')
        
        logging.info('H5 model saved successfully')
    except Exception as e:
        logging.error(f'Error saving H5 model: {str(e)}')
        raise

    logging.info(f'Saving and converting TFLite model to: {tflite_path}')
    try:
        input_shape = model.inputs[0].shape
        concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec(input_shape, tf.float32))
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  
            tf.lite.OpsSet.SELECT_TF_OPS     
]
        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        if tflite_path.exists():
            size = tflite_path.stat().st_size
            logging.info(f'TFLite model saved successfully. File size: {size/1024/1024:.2f} MB')
        else:
            raise FileNotFoundError(f'TFLite model file not found after saving: {tflite_path}')

        logging.info('TFLite model saved successfully')
    except Exception as e:
        logging.error(f'Error saving TFlite model: {str(e)}')
        raise

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
                metadata_path=Config.METADATA_PATH,
                image_dir=Config.IMAGES_DIR,
                imbalance_factor=Config.IMBALANCE_FACTOR,
                batch_size=Config.BATCH_SIZE
            )
            
            model = create_mobile_model(num_classes=1)
            
            logging.info('Starting training...')
            history = train_model(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                checkpoint_dir=output_dir / 'checkpoints',
                model_name='skin_cancer_model',
                epochs=Config.EPOCHS,
                initial_learning_rate=Config.LEARNING_RATE
            )
            
            logging.info('Evaluating model...')
            evaluate_model(model, test_dataset)
            
            logging.info('Saving models...')
            save_models(
                    model,
                    Config.MODEL_DIR / 'skc_model.h5',
                    Config.MODEL_DIR / 'skc_model.tflite'
                    )
            
            logging.info('Training completed successfully!')
            
    except Exception as e:
        logging.error(f'Error during execution: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()