import os
from roboflow import Roboflow
from tqdm import tqdm
import argparse

def main(args):
    # --- 1. Configuration ---
    # The API key is taken from the command line argument.
    # NEVER hardcode your API key in the script.
    API_KEY = args.api_key
    WORKSPACE_ID = "sirius-yhut7"
    PROJECT_ID = "sirius-ewozz"
    UPLOAD_FOLDER = "data/roboflow_upload"

    # --- 2. Initialize Roboflow ---
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    except Exception as e:
        print(f"Error initializing Roboflow. Please check your API key and project details.")
        print(e)
        return

    # --- 3. Find Files and Upload ---
    # Get all image files, ignoring labels for this pass
    image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images to upload.")

    for image_name in tqdm(image_files, desc="Uploading to Roboflow"):
        image_path = os.path.join(UPLOAD_FOLDER, image_name)
        
        # The corresponding annotation file has the same name but with a .txt extension
        annotation_name = os.path.splitext(image_name)[0] + ".txt"
        annotation_path = os.path.join(UPLOAD_FOLDER, annotation_name)

        try:
            # The upload function handles both the image and its annotation
            project.upload(
                image_path=image_path,
                annotation_path=annotation_path,
                split="train", # Upload all images to the 'train' set for now
                batch_name="pre-annotated-batch-1" # Group them in Roboflow
            )
        except Exception as e:
            print(f"\nError uploading {image_name}: {e}")
            continue

    print("\nUpload complete!")
    print("Please check your Roboflow project to see the uploaded images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a dataset with annotations to Roboflow.")
    parser.add_argument("--api-key", required=True, type=str, help="Your Roboflow private API key.")
    
    args = parser.parse_args()
    main(args)
