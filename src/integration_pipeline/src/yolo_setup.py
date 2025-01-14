import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

class YOLOProcessor:
    def __init__(self, yolo_model_path, conf_threshold=0.2, imgsz=608):
        self.model = YOLO(yolo_model_path, task='detect')
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.output_dir = Path("cropped_detections")
        self.output_dir.mkdir(exist_ok=True)

    def process_single_image(self, img_path, save_crops=False):
        try:
            results = self.model.predict(
                source=str(img_path),
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                save=False,
                device='cpu',
                verbose=False
            )
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                print(f"Could not read image: {img_path}")
                return None
            
            img_height, img_width = original_img.shape[:2]

            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    confidences = [box.conf.item() for box in boxes]
                    best_idx = np.argmax(confidences)
                    box = boxes[best_idx]
                    
                    if box.conf.item() > self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        padding = int((x2 - x1) * 0.2)
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = max(0, x2 + padding)
                        y2 = max(0, y2 + padding)

                        class_id = box.cls.item()

                        cropped_img = original_img[y1:y2, x1:x2]
                        
                        output_stem = f"{Path(img_path).stem}_det_cls{int(class_id)}_conf{box.conf.item():.2f}"
                        output_path = self.output_dir / f"{output_stem}_original.jpg"
                        cv2.imwrite(str(output_path), cropped_img)
                        
                        return str(output_path)
            return None 

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

    # def process_batch(self, image_paths, save_crops=False):
    #     results = {}
    #     total_images = len(image_paths)
        
    #     print(f"\nProcessing {total_images} images with YOLO...")
        
    #     for i, img_path in enumerate(image_paths):
    #         if (i + 1) % 100 == 0:
    #             print(f"Processed {i+1}/{total_images} images")
                
    #         cropped_img = self.process_single_image(img_path, save_crops)
    #         if cropped_img is not None:
    #             results[img_path] = cropped_img
        
    #     print(f"\nYOLO processing completed. Successfully processed {len(results)}/{total_images} images")
    #     return results