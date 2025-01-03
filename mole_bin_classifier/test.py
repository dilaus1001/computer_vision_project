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
    MODEL_PATH = 'outputs/skc_model.h5'
    DATA_DIR = '/Users/francesco/Repository/computer_vision_project/dataset/mix_dataset/balanced_images'
    RESULTS_DIR = 'outputs/test_results'

    BATCH_SIZE = 32
    PLOT_SIZE = (10, 8)
    
def setup_logging(output_dir: Path):
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'testing.log'),
            logging.StreamHandler()
        ]
    )

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
    
    # Convert to numpy arrays
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
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(report).transpose()
    df.to_csv(save_path)
    return df

def main():
    """
    1. Setting up the environment
    2. Loading the model and test data
    3. Making predictions
    4. Analyzing and visualizing results
    """
    results_dir = Path(TestConfig.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(results_dir)
    logging.info('Starting model testing...')
    
    try:
        logging.info('Loading test dataset...')
        _, _, test_dataset = prepare_datasets(
            data_dir=TestConfig.DATA_DIR,
            batch_size=TestConfig.BATCH_SIZE
        )
  
        logging.info('Loading trained model...')
        model = tf.keras.models.load_model(TestConfig.MODEL_PATH)
    
        logging.info('Making predictions...')
        y_true, y_pred, y_pred_proba = analyze_predictions(model, test_dataset)

        logging.info('Generating visualizations...')
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