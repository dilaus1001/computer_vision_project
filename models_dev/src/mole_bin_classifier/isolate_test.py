import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import pandas as pd

from model import create_mobile_model
from preprocess import prepare_datasets

class TestConfig:
    # Match the paths with your main configuration
    INPUT_DIR = Path('/Users/francesco/repos/computer_vision_project/dataset/mix_dataset')
    MODEL_DIR = Path('/Users/francesco/Repository/computer_vision_project/saved_models')
    OUTPUT_DIR = Path('outputs/test_results')
    
    METADATA_PATH = INPUT_DIR / 'mix_dataset.csv'
    IMAGES_DIR = INPUT_DIR / 'all_images'
    MODEL_PATH = MODEL_DIR / 'skc_model.h5'
    
    BATCH_SIZE = 32
    PLOT_SIZE = (10, 8)
    IMBALANCE_FACTOR = 1.2  # Match with your training config

def setup_logging(output_dir: Path):
    """Set up logging with both file and console handlers"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_dir / 'testing.log')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # Test logging setup
    logging.info(f'Testing log initialized. Output directory: {output_dir}')

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=TestConfig.PLOT_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=TestConfig.PLOT_SIZE)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def analyze_predictions(model, test_dataset):
    all_predictions = []
    all_labels = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        all_predictions.extend(predictions.flatten())
        all_labels.extend(labels.numpy().flatten())
    
    y_pred_proba = np.array(all_predictions)
    y_true = np.array(all_labels)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return y_true, y_pred, y_pred_proba

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(
        y_true, 
        y_pred,
        target_names=['Benign', 'Malignant'],
        output_dict=True
    )
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path)
    return df

def main():
    """Main testing pipeline"""
    results_dir = TestConfig.OUTPUT_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(results_dir)
    logging.info('='*50)
    logging.info('Starting model testing...')
    
    try:
        logging.info('Loading test dataset...')
        _, _, test_dataset = prepare_datasets(
            metadata_path=TestConfig.METADATA_PATH,
            image_dir=TestConfig.IMAGES_DIR,
            imbalance_factor=TestConfig.IMBALANCE_FACTOR,
            batch_size=TestConfig.BATCH_SIZE
        )
        
        logging.info('Loading trained model...')
        model = tf.keras.models.load_model(TestConfig.MODEL_PATH)
        
        logging.info('Making predictions...')
        y_true, y_pred, y_pred_proba = analyze_predictions(model, test_dataset)
        
        logging.info('Generating visualizations and reports...')
        plot_confusion_matrix(
            y_true, y_pred,
            save_path=results_dir / 'confusion_matrix.png'
        )
        
        roc_auc = plot_roc_curve(
            y_true, y_pred_proba,
            save_path=results_dir / 'roc_curve.png'
        )
        
        report_df = save_classification_report(
            y_true, y_pred,
            save_path=results_dir / 'classification_report.csv'
        )
        
        logging.info(f'ROC AUC Score: {roc_auc:.4f}')
        logging.info('\nClassification Report:')
        logging.info('\n' + str(report_df))
        logging.info('Testing completed successfully!')
        
    except Exception as e:
        logging.error(f'Error during testing: {str(e)}', exc_info=True)
        raise

if __name__ == '__main__':
    main()