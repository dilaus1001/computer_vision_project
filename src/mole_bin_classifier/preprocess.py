import tensorflow as tf
import numpy as np
import os
import shutil
from pathlib import Path
from sklearn.utils import resample, shuffle
from sklearn.model_selection import train_test_split
import logging
from PIL import Image
from typing import Tuple, List
from ultralytics import YOLO
import pandas as pd

class SkinLesionDataset:
    def __init__(self, metadata_df: pd.DataFrame, is_training: bool = False):
        self.image_paths = metadata_df['image_path'].tolist()
        self.labels = metadata_df['label'].tolist()
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

        angle = tf.random.uniform([], -0.2, 0.2)
        image = tf.image.rot90(image, k=tf.cast(angle / (np.pi/2), tf.int32))
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

def prepare_datasets(metadata_path: str, image_dir: str, imbalance_factor=1.2, random_state=123, batch_size=32):
    df = pd.read_csv(metadata_path)
    image_dir = Path(image_dir)

    # (0: benign, 1: malignant)
    df['label'] = (df['benign_malignant'] == 'malignant').astype(int)
    df['image_path'] = df['image_name'].apply(lambda x: str(image_dir / f"{x}.jpg"))
    df = df[df['image_path'].apply(lambda x: Path(x).exists())]

    benign_samples = df[df['label'] == 0]
    malignant_samples = df[df['label'] == 1]

    logging.info(f"Original distribution:")
    logging.info(f"Benign samples: {len(benign_samples)}")
    logging.info(f"Malignant samples: {len(malignant_samples)}")

    target_malignant = len(malignant_samples)
    target_benign = int(target_malignant * imbalance_factor)

    if len(benign_samples) > target_benign:
        benign_reduced = benign_samples.sample(n=target_benign, random_state=random_state)
    else:
        benign_reduced = benign_samples

    balanced_df = pd.concat([benign_reduced, malignant_samples]).reset_index(drop=True)
    balanced_df = shuffle(balanced_df, random_state=random_state)

    logging.info(f"Reduced distribution with slight imbalance (factor={imbalance_factor}):")
    logging.info(balanced_df['label'].value_counts())

    train_df, temp_df = train_test_split(
        balanced_df, 
        test_size=0.4, 
        stratify=balanced_df['label'], 
        random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        stratify=temp_df['label'], 
        random_state=random_state
    )
    train_dataset = SkinLesionDataset(
        train_df, 
        is_training=True
    )
    val_dataset = SkinLesionDataset(
        val_df,
        is_training=False
    )
    test_dataset = SkinLesionDataset(
        test_df,  
        is_training=False
    )

    return (
        train_dataset.create_dataset(batch_size=batch_size, shuffle=True),
        val_dataset.create_dataset(batch_size=batch_size),
        test_dataset.create_dataset(batch_size=batch_size)
    )

