from ultralytics import YOLO
import os

def main():
    # --- 1. Load the model ---
    # We are now starting from the best model we trained on synthetic data.
    # This is called fine-tuning and helps the model learn faster and better.
    previous_best_model = 'runs/train3/weights/best.pt'
    
    if os.path.exists(previous_best_model):
        model = YOLO(previous_best_model)
        print(f"Starting fine-tuning from: {previous_best_model}")
    else:
        # Fallback to a generic pretrained model if the previous one isn't found
        model = YOLO('yolov8n.pt')
        print("Previous model not found. Starting from generic yolov8n.pt.")

    # --- 2. Train the model ---
    # We now point the 'data' argument to our new, human-verified Roboflow dataset.
    # We increase the epochs to 100 for a more thorough training run on this high-quality data.
    results = model.train(
        data='roboflow/data.yaml', 
        epochs=100, 
        imgsz=640, 
        patience=10, 
        project='runs',
        name='final_model' # Give this run a distinct name
    )

    print("Final training finished.")
    print("New, improved model and results saved in the 'runs/final_model' directory.")

if __name__ == '__main__':
    main()
