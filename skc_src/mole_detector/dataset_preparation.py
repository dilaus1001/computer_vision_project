import os
import shutil
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

class HAM10000Processor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / 'all_images'
        self.masks_dir = self.base_dir / 'HAM10000_segmentations_lesion_tschandl'
        self.metadata_path = self.base_dir / 'HAM10000_metadata.csv'
        
        # Create YOLO directory structure
        self.yolo_dir = self.base_dir / 'yolo_dataset'
        self.train_dir = self.yolo_dir / 'train'
        self.val_dir = self.yolo_dir / 'val'
        self.test_dir = self.yolo_dir / 'test'

    def create_directory_structure(self):
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            (dir_path / 'images').mkdir(parents=True, exist_ok=True)
            (dir_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        print("Created directory structure")

    def mask_to_bbox(self, mask_path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        image_h, image_w = mask.shape
        x_center = (x + w/2) / image_w
        y_center = (y + h/2) / image_h
        width = w / image_w
        height = h / image_h
        
        return (x_center, y_center, width, height)

    def process_dataset(self, train_size=0.7, val_size=0.15, test_size=0.15):
        image_files = list(self.images_dir.glob('*.jpg'))
        
        train_val_files, test_files = train_test_split(image_files, 
                                                      test_size=test_size, 
                                                      random_state=42)
        
        val_size_adjusted = val_size / (train_size + val_size)
        train_files, val_files = train_test_split(train_val_files, 
                                                 test_size=val_size_adjusted, 
                                                 random_state=42)
        
        # Process each split
        self._process_split(train_files, 'train')
        self._process_split(val_files, 'val')
        self._process_split(test_files, 'test')
        
        # Create dataset.yaml
        self._create_dataset_yaml()
        
        print(f"Processed {len(train_files)} training images")
        print(f"Processed {len(val_files)} validation images")
        print(f"Processed {len(test_files)} test images")

    def _process_split(self, image_files, split_type):
        split_dir = getattr(self, f'{split_type}_dir')
        
        for img_path in image_files:
            mask_path = self.masks_dir / f'{img_path.stem}_segmentation.png'
            if not mask_path.exists():
                print(f"Warning: No mask found for {img_path.name}")
                continue
            
            # Get bounding box
            bbox = self.mask_to_bbox(mask_path)
            if bbox is None:
                print(f"Warning: Could not process mask for {img_path.name}")
                continue

            shutil.copy2(img_path, split_dir / 'images' / img_path.name)
            
            # Save YOLO format annotation
            label_path = split_dir / 'labels' / f'{img_path.stem}.txt'
            with open(label_path, 'w') as f:
                x_center, y_center, width, height = bbox
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _create_dataset_yaml(self):
        yaml_content = f"""path: {self.yolo_dir.absolute()}
train: train/images
val: val/images
test: test/images

names:
    0: lesion
"""
        with open(self.yolo_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)

    def verify_random_samples(self, num_samples=5):
        for split in ['train', 'val', 'test']:
            split_dir = getattr(self, f'{split}_dir')
            image_files = list((split_dir / 'images').glob('*.jpg'))
            if not image_files:
                continue

            sample_files = np.random.choice(image_files, 
                                          size=min(num_samples, len(image_files)), 
                                          replace=False)
            
            for img_path in sample_files:
                image = cv2.imread(str(img_path))
                h, w = image.shape[:2]
                
                label_path = split_dir / 'labels' / f'{img_path.stem}.txt'
                if not label_path.exists():
                    continue
                    
                with open(label_path, 'r') as f:
                    line = f.readline().strip()
                    _, x_center, y_center, width, height = map(float, line.split())
                
                x = int((x_center - width/2) * w)
                y = int((y_center - height/2) * h)
                box_w = int(width * w)
                box_h = int(height * h)

                cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                
                cv2.imshow(f'Verification - {split}', image)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = HAM10000Processor('/Users/francesco/Repository/computer_vision_project/dataset/HAM10000')
    processor.create_directory_structure()
    processor.process_dataset()
    processor.verify_random_samples()