import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from yolo_setup import YOLOProcessor
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, skc_model_path, yolo_model_path, dataset_csv_path, img_dir, target_size=(224,224), balance_factor=1.2, reduction_factor=1):
        self.skc_model_path = skc_model_path
        self.yolo_model_path = yolo_model_path
        self.img_dir = Path(img_dir)
        self.target_size = target_size
        self.balance_factor = balance_factor
        self.reduction_factor = reduction_factor

        self.dataset_df = self.prepare_balanced_dataset(dataset_csv_path)
        self.setup_model()
        self.yolo_processor = YOLOProcessor(self.yolo_model_path, conf_threshold=0.4)

        self.files_not_found = 0
        self.files_processed = 0

    def setup_model(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.skc_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def preprocess_image(self, img_path):
        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize_with_pad(img, self.target_size[0], self.target_size[1])
        return img.numpy()
    
    def prepare_balanced_dataset(self, dataset_csv_path):
        df = pd.read_csv(dataset_csv_path)

        malignant_df = df[df['benign_malignant'] == 'malignant']
        benign_df = df[df['benign_malignant'] == 'benign']

        n_malignant = len(malignant_df)
        target_benign = int(n_malignant * self.balance_factor)
        print(f"\nOriginal distribution:")
        print(f"Malignant samples: {n_malignant}")
        print(f"Benign samples: {len(benign_df)}")

        # Dataset balancing
        if len(benign_df) > target_benign:
            benign_df = benign_df.sample(n=target_benign, random_state=123)
        balanced_df = pd.concat([malignant_df, benign_df])

        # Dataset reduction
        if self.reduction_factor < 1.0:
            total_samples = len(balanced_df)
            target_samples = int(total_samples * self.reduction_factor)
            balanced_df = balanced_df.sample(n=target_samples, random_state=123)

        final_malignant = len(balanced_df[balanced_df['benign_malignant'] == 'malignant'])
        final_benign = len(balanced_df[balanced_df['benign_malignant'] == 'benign'])
        print(f"\nFinal distribution after balancing and reduction:")
        print(f"Malignant samples: {final_malignant}")
        print(f"Benign samples: {final_benign}")
        print(f"Total samples: {len(balanced_df)}")

        return balanced_df

    def single_predict(self, img_path, conf_factor=0.5, use_yolo=False):
        try:
            if not Path(img_path).exists():
                self.files_not_found += 1
                return None, None
                
            if use_yolo:
                cropped_img_path = self.yolo_processor.process_single_image(img_path)
                if cropped_img_path is None:
                    print(f"YOLO processing failed for {img_path}")
                    return None, None
                processed_img = self.preprocess_image(cropped_img_path)
            else:
                processed_img = self.preprocess_image(img_path)
            
            input_data = np.expand_dims(processed_img, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction_value = output_data[0][0]
            
            self.files_processed += 1
            return prediction_value > conf_factor, prediction_value
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None, None
    
    def evaluate_model(self, conf_factor=0.5, use_yolo=False):
        y_true = []
        y_pred = []
        confidence_scores = []

        self.files_not_found = 0
        self.files_processed = 0

        total_images = len(self.dataset_df)
        desc = 'Evaluating Full Pipeline' if use_yolo else 'Evaluating SKC Model Only'
        #pbar = tqdm(self.dataset_df.iterrows(), total=total_images, desc=desc)
        with tqdm(total=total_images, desc=desc, position=0, leave=True) as pbar:
            for _, row in self.dataset_df.iterrows():
                img_path = self.img_dir / f"{row['image_name']}.jpg"

                true_label = 1 if row['benign_malignant'] == 'malignant' else 0
                pred_label, conf_score = self.single_predict(img_path, conf_factor, use_yolo)

                if pred_label is not None:
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    confidence_scores.append(conf_score)
                    self.files_processed += 1

                pbar.update(1)
                if self.files_processed % 10 == 0:
                    pbar.set_postfix({
                        'not_found': self.files_not_found,
                        'processed': self.files_processed
                    }, refresh=True)
        
        print("\nFinal Statistics:")
        print(f"Total images in dataset: {total_images}")
        print(f"Files not found: {self.files_not_found}")
        print(f"Successfully processed: {self.files_processed}")
        print(f"Success rate: {(self.files_processed/total_images)*100:.2f}%")

        metrics = self.calculate_metrics(y_true, y_pred, confidence_scores)
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, confidence_scores):
        if not y_true or not y_pred:
            return {
                'accuracy': 0.0,
                'confusion_matrix': np.array([[0, 0], [0, 0]]),
                'classification_report': "No predictions made",
                'confidence_scores': []
            }
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred),
            'confidence_scores': confidence_scores
        }
        return metrics
    
    def plot_confusion_matrix(self, confusion_mat, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'confusion_matrix_{title.lower().replace(" ", "_")}.png')
        plt.close()

