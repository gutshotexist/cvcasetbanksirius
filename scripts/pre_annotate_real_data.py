import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
import argparse

def main(args):
    # --- 1. Load the Trained Model ---
    # We load the 'best.pt' file which contains the weights of our best-performing model.
    model_path = os.path.join(args.run_path, 'weights/best.pt')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure the path to the training run is correct.")
        return
        
    model = YOLO(model_path)
    
    # --- 2. Define Input and Output Directories ---
    input_dir = 'data/raw_images'
    output_dir_images = 'data/final_predictions/images'
    output_dir_labels = 'data/final_predictions/labels'
    
    # --- 3. Create Output Directories ---
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)
    
    # --- 4. Get List of Images to Process ---
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to pre-annotate.")
    
    # --- 5. Run Inference and Save Results ---
    for image_name in tqdm(image_files, desc="Pre-annotating images"):
        image_path = os.path.join(input_dir, image_name)
        
        # The model.predict() function runs the inference.
        # We set a confidence threshold to filter out weak detections.
        results = model.predict(image_path, conf=args.conf_threshold, verbose=False)
        
        # The result for the first (and only) image
        result = results[0]
        
        # If the model detected any objects
        if len(result.boxes) > 0:
            # Prepare the output label file path
            base_name = os.path.splitext(image_name)[0]
            label_name = f"{base_name}.txt"
            label_path = os.path.join(output_dir_labels, label_name)
            
            with open(label_path, 'w') as f:
                # result.boxes.xywhn contains the bounding boxes in normalized YOLO format
                for box in result.boxes.xywhn:
                    # The class is always 0 for our single-class model
                    class_id = 0
                    x_center, y_center, width, height = box.tolist()
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            # For simplicity, we can copy the image later or handle it in the next step.
            # For now, we'll just focus on generating the labels.

    print("\nPre-annotation finished.")
    print(f"Labels saved to: {output_dir_labels}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-annotate a dataset using a trained YOLOv8 model.")
    parser.add_argument("--run-path", type=str, default="runs/final_model", help="Path to the YOLO training run directory (e.g., 'runs/final_model').")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for object detection.")
    
    args = parser.parse_args()
    main(args)
