import tensorflow as tf
import numpy as np
import os
import shutil
import pathlib
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import logging
from PIL import Image
from typing import Tuple, List

class SkinLesionDataset:
    def __init__(self, image_paths: List[str], labels: List[int], is_training: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training

    def _augment(self, image):
        # Random flips
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image

    def _load_and_preprocess_image(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize_with_pad(img, 224, 224)
        if self.is_training:
            img = self._augment(img)
        
        return img

    def create_dataset(self, batch_size: int, shuffle: bool = False):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(self.image_paths),
                reshuffle_each_iteration=True
            )

        dataset = dataset.map(
            lambda path, label: (
                self._load_and_preprocess_image(path),
                tf.cast(label, tf.float32)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def prepare_datasets(data_dir: str, batch_size: int = 32):
    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    class_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        class_paths = list(class_dir.glob('*.[jp][pn][g]'))
        if not class_paths:
            logging.warning(f"No images found in class directory: {class_dir}")
        image_paths.extend([str(path) for path in class_paths])
        labels.extend([class_idx] * len(class_paths))
    
    if not image_paths:
        raise ValueError(
            f"No valid images found in any class directory under {data_dir}. "
            "Please ensure the directory structure is correct and contains .jpg, .jpeg, or .png files."
        )
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.4, stratify=labels, random_state=42
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create datasets
    train_dataset = SkinLesionDataset(
        train_paths, train_labels, is_training=True
    ).create_dataset(batch_size, shuffle=True)
    
    val_dataset = SkinLesionDataset(
        val_paths, val_labels, is_training=False
    ).create_dataset(batch_size)
    
    test_dataset = SkinLesionDataset(
        test_paths, test_labels, is_training=False
    ).create_dataset(batch_size)
    
    logging.info(f'Train dataset size: {len(train_paths)}')
    logging.info(f'Validation dataset size: {len(val_paths)}')
    logging.info(f'Test dataset size: {len(test_paths)}')
    
    return train_dataset, val_dataset, test_dataset