import os
from PIL import Image
import pillow_avif


def convert_avif_to_png(input_dir):
    # Loop through all directories and files in the input directory
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.avif'):
                try:
                    # Full path to the .avif file
                    avif_path = os.path.join(subdir, file)

                    # Open the .avif image
                    img = Image.open(avif_path)

                    # Define the output path (replacing .avif with .png)
                    output_path = os.path.join(subdir, file.replace('.avif', '.png'))

                    # Save the image as .png
                    img.save(output_path, format='PNG')

                    print(f"Converted: {avif_path} -> {output_path}")

                except Exception as e:
                    print(f"Error converting {file}: {e}")


# Example usage:
input_dir = "/home/toujlakh/Projects/product_classification/data"
convert_avif_to_png(input_dir)
