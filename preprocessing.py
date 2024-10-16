import os
from PIL import Image
import numpy as np

# Directories
input_dir = "/home/toujlakh/Projects/product_classification/data"
output_dir = "/home/toujlakh/Projects/product_classification/preprocessed_data"

# Target size for image resizing
target_size = (224, 224)

def preprocess_png_images(input_dir, output_dir, target_size):
    # Loop through all directories and files in the input directory
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):  # Only process .png files
                try:
                    # Full path to the .png file
                    image_path = os.path.join(subdir, file)

                    # Open the .png image
                    img = Image.open(image_path)

                    # Convert to RGB format (if not already in RGB)
                    img = img.convert("RGB")

                    # Resize the image to the target size
                    img = img.resize(target_size)

                    # Convert the image to a numpy array
                    img_array = np.array(img)

                    # Normalize pixel values to the range [0, 1]
                    img_array = img_array / 255.0

                    # Convert back to image format from the normalized array
                    img = Image.fromarray((img_array * 255).astype(np.uint8))

                    # Prepare the output path, maintaining the directory structure
                    rel_path = os.path.relpath(subdir, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)

                    # Create the output subdirectory if it doesn't exist
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # Save the preprocessed image in the output directory
                    output_image_path = os.path.join(output_subdir, file)
                    img.save(output_image_path, format='PNG')

                    print(f"Processed and saved {output_image_path}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Example usage:
preprocess_png_images(input_dir, output_dir, target_size)