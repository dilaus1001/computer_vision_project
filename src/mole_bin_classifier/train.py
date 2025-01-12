import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

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

def create_callbacks(checkpoint_dir: Path, patience: int = 8):
    """
    - Model checkpointing
    - Early stopping
    - TensorBoard logging
    - Learning rate reduction on plateau
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'model_epoch_{epoch:02d}.weights.h5'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
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
            factor=0.5,
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
                epochs=100, initial_learning_rate=0.001):
    """
    - Model compilation
    - Setting up callbacks
    - Training the model
    - Plotting training history
    """
    logging.info('Starting model training...')
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['binary_accuracy']
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

def evaluate_model(model, test_dataset):
    logging.info('Evaluating model on test dataset...')
    
    results = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True
    )
    
    logging.info('Test Results:')
    for metric_name, value in results.items():
        logging.info(f'{metric_name}: {value:.4f}')