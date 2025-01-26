import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0
        self.best = float('inf')
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current is None:
            return
            
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    logging.info(f'Restoring model weights from epoch {self.best_epoch}')
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info(f'Early stopping triggered at epoch {self.stopped_epoch}')

def create_callbacks(checkpoint_dir: Path, model_name: str = 'best_model', patience: int = 15):
    """
    - Model checkpointing
    - Early stopping
    - TensorBoard logging
    - Learning rate reduction on plateau
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = checkpoint_dir / f'{model_name}_best.h5'
    checkpoint_pattern = checkpoint_dir / f'{model_name}_epoch_{{epoch:02d}}_val_loss{{val_loss:.4f}}.h5'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_pattern),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(checkpoint_dir / 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=patience // 2,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    return callbacks

def plot_training_history(history, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax1.plot(history.history['binary_accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_model(model, train_dataset, val_dataset, checkpoint_dir='checkpoints', 
                model_name='skin_cancer_model', epochs=100, initial_learning_rate=0.001):
    """
    - Model compilation
    - Setting up callbacks
    - Training the model
    - Plotting training history
    """
    logging.info('Starting model training...')
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    metrics = [
        'binary_accuracy',
        tf.keras.metrics.AUC(name='auc'),                    # ROC AUC
        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),     # Precision-Recall AUC
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.BinaryAccuracy(name='binary_acc', threshold=0.5)
    ]

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )

    callbacks = create_callbacks(Path(checkpoint_dir))
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=str(Path(checkpoint_dir) / 'training_history.png')
    )
    
    return history

def evaluate_model(model, test_dataset, output_dir='outputs/test_results'):
    logging.info('Starting model evaluation...')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model predictions
    all_predictions = []
    all_labels = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        all_predictions.extend(predictions.flatten())
        all_labels.extend(labels.numpy().flatten())
    
    y_pred_proba = np.array(all_predictions)
    y_true = np.array(all_labels)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Basic metrics evaluation
    results = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True
    )
    
    logging.info('\nTest Metrics:')
    for metric_name, value in results.items():
        logging.info(f'{metric_name}: {value:.4f}')
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(output_dir / 'precision_recall_curve.png')
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred_proba, bins=50, kde=True)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'prediction_distribution.png')
    plt.close()
    
    # Log additional metrics
    logging.info(f'\nAdditional Metrics:')
    logging.info(f'ROC AUC Score: {roc_auc:.4f}')
    logging.info(f'PR AUC Score: {pr_auc:.4f}')
    
    # Calculate class-wise metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    
    logging.info(f'\nClass-wise Metrics:')
    logging.info(f'Sensitivity (True Positive Rate): {sensitivity:.4f}')
    logging.info(f'Specificity (True Negative Rate): {specificity:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    
    return {
        'basic_metrics': results,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }