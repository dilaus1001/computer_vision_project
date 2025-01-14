from model_evaluator import ModelEvaluator

def main():
    skc_model_path = '/Users/francesco/repos/computer_vision_project/saved_models/skc_model.tflite'
    yolo_model_path = '/Users/francesco/repos/computer_vision_project/saved_models/yolo_trained/yolo11n_float16.tflite'
    dataset_csv_path = '/Users/francesco/repos/computer_vision_project/dataset/mix_dataset/mix_dataset.csv'
    #image_dir = '/Users/francesco/repos/computer_vision_project/dataset/mix_dataset/train_images'
    image_dir = '/Users/francesco/repos/computer_vision_project/dataset/mix_dataset/tobedeletemix'

    conf_factor = 0.5
    evaluator = ModelEvaluator(skc_model_path, yolo_model_path, dataset_csv_path, image_dir)

    print("\nEvaluate skc Model Only...")
    skc_model_metrics = evaluator.evaluate_model(conf_factor=conf_factor, use_yolo=False)
    print(f"\nskc Model Only Accuracy: {skc_model_metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(skc_model_metrics['classification_report'])
    evaluator.plot_confusion_matrix(skc_model_metrics['confusion_matrix'], 'skc Model')

    print("\nEvaluating Full Pipeline (with YOLO)...")
    pipeline_metrics = evaluator.evaluate_model(conf_factor=conf_factor, use_yolo=True)
    print(f"\nFull Pipeline Accuracy: {pipeline_metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(pipeline_metrics['classification_report'])
    evaluator.plot_confusion_matrix(pipeline_metrics['confusion_matrix'], 'Full Pipeline')

    print("\nAccuracy Comparison:")
    print(f"Custom Model Only: {skc_model_metrics['accuracy']:.4f}")
    print(f"Full Pipeline: {pipeline_metrics['accuracy']:.4f}")
    print(f"Accuracy Difference: {(pipeline_metrics['accuracy'] - skc_model_metrics['accuracy']):.4f}")

if __name__ == "__main__":
    main()