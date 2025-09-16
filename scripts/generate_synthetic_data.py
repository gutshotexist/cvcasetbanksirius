import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import random
import argparse
from tqdm import tqdm

def create_output_dirs(output_path):
    """Creates the necessary directory structure for the YOLO dataset."""
    os.makedirs(os.path.join(output_path, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels/train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels/val'), exist_ok=True)

def paste_transparent(background, foreground, box):
    """
    Pastes a foreground PIL image with transparency onto a background PIL image.
    `box` is a (left, top) tuple for the paste location.
    """
    # Extract the alpha channel from the foreground and paste it
    background.paste(foreground, box, foreground)
    return background

def create_composite_logo(shield_img, letter_img):
    """
    Creates a composite logo by placing the letter inside the shield.
    Returns a single PIL image of the composite logo.
    """
    # Resize letter to be smaller than the shield (e.g., 70% of shield's smallest dimension)
    shield_w, shield_h = shield_img.size
    scale = 0.7
    new_letter_size = int(min(shield_w, shield_h) * scale)
    
    letter_img = letter_img.resize((new_letter_size, new_letter_size), Image.Resampling.LANCZOS)
    letter_w, letter_h = letter_img.size

    # Create a new transparent image to hold the composite logo
    composite = Image.new('RGBA', shield_img.size, (0, 0, 0, 0))
    composite = paste_transparent(composite, shield_img, (0, 0))

    # Position letter in the center of the shield
    paste_x = (shield_w - letter_w) // 2
    paste_y = (shield_h - letter_h) // 2
    composite = paste_transparent(composite, letter_img, (paste_x, paste_y))
    
    return composite

def apply_augmentations(logo_img):
    """
    Applies a series of augmentations to the logo image.
    Includes perspective warp, rotation, and color jitter.
    """
    # --- 1. Perspective Warp (using OpenCV) ---
    logo_cv = cv2.cvtColor(np.array(logo_img), cv2.COLOR_RGBA2BGRA)
    h, w = logo_cv.shape[:2]

    # Define original corners
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Define distorted corners by adding random jitter
    max_jitter = 0.15 # Jitter up to 15% of the image dimension
    jitter_w = int(w * max_jitter)
    jitter_h = int(h * max_jitter)
    
    dst_pts = np.float32([
        [random.randint(0, jitter_w), random.randint(0, jitter_h)],
        [w - random.randint(1, jitter_w), random.randint(0, jitter_h)],
        [w - random.randint(1, jitter_w), h - random.randint(1, jitter_h)],
        [random.randint(0, jitter_w), h - random.randint(1, jitter_h)]
    ])
    
    # Get the perspective transformation matrix and warp the image
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_logo_cv = cv2.warpPerspective(logo_cv, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # Convert back to PIL Image
    logo_img = Image.fromarray(cv2.cvtColor(warped_logo_cv, cv2.COLOR_BGRA2RGBA))

    # --- 2. Rotation ---
    angle = random.uniform(-15, 15)
    logo_img = logo_img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # --- 3. Brightness & Contrast ---
    enhancer = ImageEnhance.Brightness(logo_img)
    logo_img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    enhancer = ImageEnhance.Contrast(logo_img)
    logo_img = enhancer.enhance(random.uniform(0.8, 1.2))

    return logo_img

def convert_to_yolo(box, img_size):
    """Converts a bounding box (xmin, ymin, xmax, ymax) to YOLO format."""
    img_w, img_h = img_size
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    x_center /= img_w
    w /= img_w
    y_center /= img_h
    h /= img_h
    
    return (x_center, y_center, w, h)

def main(args):
    # --- 1. Define Paths ---
    asset_path = 'data/synthetic_assets'
    shield_path = os.path.join(asset_path, 'shields')
    letter_path = os.path.join(asset_path, 'letters')
    background_path = os.path.join(asset_path, 'backgrounds')
    output_path = 'data/synthetic_dataset'

    # --- 2. Create Output Directories ---
    create_output_dirs(output_path)

    # --- 3. Load Asset File Paths ---
    shield_files = [os.path.join(shield_path, f) for f in os.listdir(shield_path) if f.endswith(('.png', '.jpg'))]
    letter_files = [os.path.join(letter_path, f) for f in os.listdir(letter_path) if f.endswith(('.png', '.jpg'))]
    background_files = [os.path.join(background_path, f) for f in os.listdir(background_path) if f.endswith(('.png', '.jpg'))]

    # Check if we have assets
    if not all([shield_files, letter_files, background_files]):
        print("Error: One of the asset folders is empty. Please populate 'shields', 'letters', and 'backgrounds'.")
        return

    print(f"Found {len(shield_files)} shields, {len(letter_files)} letters, and {len(background_files)} backgrounds.")
    print(f"Generating {args.num_images} synthetic images...")

    # --- Main Loop ---
    generated_count = 0
    with tqdm(total=args.num_images, desc="Generating images") as pbar:
        while generated_count < args.num_images:
            try:
                # --- Get a background ---
                background_pil = Image.open(random.choice(background_files)).convert("RGB")
                bg_w, bg_h = background_pil.size
                
                # List to store all bboxes for this image
                yolo_bboxes = []
                
                # --- Decide how many logos to add ---
                num_logos_to_add = random.randint(1, 3)

                for _ in range(num_logos_to_add):
                    # --- 4. Load Random Assets ---
                    shield_file_path = random.choice(shield_files)
                    letter_file_path = random.choice(letter_files)

                    # Get just the filenames for the check
                    shield_filename = os.path.basename(shield_file_path)
                    letter_filename = os.path.basename(letter_file_path)
                    
                    # --- Exclusion Logic ---
                    if (shield_filename == 'shield1.png' and letter_filename == 'tletter2.png') or \
                       (shield_filename == 'shield3.png' and letter_filename == 'tletter1.png'):
                        continue # Skip this forbidden combination and try a new one

                    shield_pil = Image.open(shield_file_path).convert("RGBA")
                    letter_pil = Image.open(letter_file_path).convert("RGBA")

                    # --- 5. Create Composite Logo ---
                    logo = create_composite_logo(shield_pil, letter_pil)

                    # --- 5.5 Apply Advanced Augmentations ---
                    logo = apply_augmentations(logo)

                    # --- 6. Augmentations (Resize) ---
                    min_scale = 0.05
                    max_scale = 0.35 # Reduced max size to make room for multiple logos
                    scale = random.uniform(min_scale, max_scale)
                    
                    target_w = int(bg_w * scale)
                    target_h = int(bg_h * scale)
                    
                    logo_w, logo_h = logo.size
                    # handle case where logo is 0-size
                    if logo_w == 0 or logo_h == 0: continue
                    ratio = min(target_w / logo_w, target_h / logo_h)
                    new_logo_w = int(logo_w * ratio)
                    new_logo_h = int(logo_h * ratio)

                    if new_logo_w < 1 or new_logo_h < 1:
                        continue 

                    logo = logo.resize((new_logo_w, new_logo_h), Image.Resampling.LANCZOS)
                    
                    # --- 7. Paste Logo and Get BBox ---
                    max_x = bg_w - new_logo_w
                    max_y = bg_h - new_logo_h
                    paste_x = random.randint(0, max_x)
                    paste_y = random.randint(0, max_y)
                    
                    background_pil = paste_transparent(background_pil, logo, (paste_x, paste_y))
                    
                    bbox = (paste_x, paste_y, paste_x + new_logo_w, paste_y + new_logo_h)
                    
                    # --- 8. Convert to YOLO and store ---
                    yolo_bbox = convert_to_yolo(bbox, (bg_w, bg_h))
                    yolo_bboxes.append(yolo_bbox)

                # --- 9. Save Image and Labels (if any logos were placed) ---
                if not yolo_bboxes:
                    continue # Skip saving if no logos were successfully placed

                # Determine if train or val
                split = 'train' if random.random() < 0.9 else 'val'
                
                # Save image
                img_filename = f"synthetic_{generated_count:06d}.jpg"
                background_pil.save(os.path.join(output_path, f'images/{split}', img_filename))
                
                # Save label
                label_filename = f"synthetic_{generated_count:06d}.txt"
                with open(os.path.join(output_path, f'labels/{split}', label_filename), 'w') as f:
                    for bbox in yolo_bboxes:
                        f.write(f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\\n")
                
                generated_count += 1
                pbar.update(1)

            except Exception as e:
                # Using pbar.write to not interfere with the progress bar
                pbar.write(f"\nWarning: Could not process an image. Error: {e}. Skipping.")
                continue

    print(f"\nSuccessfully generated {generated_count} images.")
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for YOLO object detection.")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of synthetic images to generate.")
    
    args = parser.parse_args()
    main(args)
