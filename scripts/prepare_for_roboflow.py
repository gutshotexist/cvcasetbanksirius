import os
import shutil
from tqdm import tqdm

def main():
    # --- 1. Define Paths ---
    raw_images_dir = 'data/raw_images'
    pre_annotated_labels_dir = 'data/pre_annotated/labels'
    output_dir = 'data/roboflow_upload'

    # --- 2. Create the output directory ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    # --- 3. Get the list of all label files ---
    label_files = [f for f in os.listdir(pre_annotated_labels_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files to process.")

    # --- 4. Copy images and labels ---
    copied_count = 0
    not_found_count = 0
    for label_filename in tqdm(label_files, desc="Preparing files for Roboflow"):
        # Derive the corresponding image filename (assuming .jpg)
        base_name = os.path.splitext(label_filename)[0]
        image_filename_jpg = f"{base_name}.jpg"
        image_filename_png = f"{base_name}.png" # Also check for png
        image_filename_jpeg = f"{base_name}.jpeg" # Also check for jpeg

        source_label_path = os.path.join(pre_annotated_labels_dir, label_filename)
        
        # Check for possible image extensions
        source_image_path = None
        if os.path.exists(os.path.join(raw_images_dir, image_filename_jpg)):
            source_image_path = os.path.join(raw_images_dir, image_filename_jpg)
        elif os.path.exists(os.path.join(raw_images_dir, image_filename_png)):
             source_image_path = os.path.join(raw_images_dir, image_filename_png)
        elif os.path.exists(os.path.join(raw_images_dir, image_filename_jpeg)):
             source_image_path = os.path.join(raw_images_dir, image_filename_jpeg)

        if source_image_path:
            # Define destination paths
            dest_image_path = os.path.join(output_dir, os.path.basename(source_image_path))
            dest_label_path = os.path.join(output_dir, label_filename)
            
            # Copy the files
            shutil.copyfile(source_image_path, dest_image_path)
            shutil.copyfile(source_label_path, dest_label_path)
            copied_count += 1
        else:
            not_found_count += 1
            # print(f"Warning: Could not find matching image for label: {label_filename}")

    print(f"\nPreparation complete.")
    print(f"Successfully copied {copied_count} images and their labels to {output_dir}")
    if not_found_count > 0:
        print(f"Warning: Could not find matching images for {not_found_count} labels.")

if __name__ == '__main__':
    main()



