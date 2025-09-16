from ultralytics import YOLO

def main():
    """
    This script evaluates the final trained model on the 'test' split of the dataset.
    The 'test' split was not seen by the model during training or validation,
    providing the most objective performance metrics.
    """
    
    # --- 1. Load the Final Trained Model ---
    model_path = 'runs/final_model/weights/best.pt'
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}")
        print("Please ensure the final training run completed successfully.")
        print(e)
        return

    # --- 2. Run Evaluation on the Test Set ---
    print("Evaluating the final model on the closed test set...")
    
    # The model.val() function can be pointed to a specific data split.
    metrics = model.val(data='roboflow/data.yaml', split='test')
    
    print("\n--- Final Model Performance on Test Set ---")
    # The metrics object contains all the performance data.
    # We are interested in mAP50, which corresponds to IoU=0.5.
    print(f"Precision (IoU=0.5): {metrics.box.p[0]:.4f}")
    print(f"Recall (IoU=0.5): {metrics.box.r[0]:.4f}")
    print(f"mAP@50 (IoU=0.5): {metrics.box.map50:.4f}")
    
    # Calculate and print F1-score
    precision = metrics.box.p[0]
    recall = metrics.box.r[0]
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"F1-Score (IoU=0.5): {f1_score:.4f}")
    else:
        print("F1-Score could not be calculated.")

if __name__ == '__main__':
    main()
