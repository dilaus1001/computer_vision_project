from yolo_setup import YOLOProcessor
from pathlib import Path
from tqdm import tqdm

def process_directory(input_dir, yolo_model_path, conf_threshold=0.2):
    """
    Process all images in a directory using the existing YOLOProcessor
    """
    # Initialize YOLO processor
    processor = YOLOProcessor(yolo_model_path, conf_threshold)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in Path(input_dir).glob('*') 
        if f.suffix.lower() in image_extensions
    ]
    
    # Process statistics
    total_images = len(image_files)
    successful = 0
    failed = 0
    
    print(f"Found {total_images} images to process")
    
    # Process each image
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for img_path in image_files:
            # Process image using existing processor
            result = processor.process_single_image(img_path)
            
            if result is not None:
                successful += 1
            else:
                failed += 1
                print(f"Failed to process: {img_path}")
            
            pbar.update(1)
            pbar.set_postfix({
                'successful': successful,
                'failed': failed
            })
    
    # Print final statistics
    print("\nProcessing completed!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/total_images)*100:.2f}%")
    print(f"Cropped images saved in: {processor.output_dir}")

def main():
    # Configuration
    input_dir = "/Users/francesco/repos/computer_vision_project/dataset/mix_dataset/all_images"  # Replace with your input directory
    yolo_model_path = "/Users/francesco/repos/computer_vision_project/src/mole_detector/yolo_train/train3/weights/best.pt"  # Replace with your YOLO model path
    conf_threshold = 0.2
    
    # Process the directory
    process_directory(
        input_dir=input_dir,
        yolo_model_path=yolo_model_path,
        conf_threshold=conf_threshold
    )

if __name__ == "__main__":
    main()