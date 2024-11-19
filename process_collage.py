from PIL import Image
import os

# File paths
input_path = r"C:\Users\Pablo\Desktop\EPFL\splineops\collagen.tif"
output_path = r"C:\Users\Pablo\Desktop\EPFL\splineops\collagen_grayscale.png"

try:
    # Open the image
    with Image.open(input_path) as img:
        # Convert to grayscale
        grayscale_img = img.convert("L")
        
        # Save as PNG
        grayscale_img.save(output_path, "PNG")
    
    print(f"Image successfully converted to grayscale and saved as PNG at: {output_path}")

except FileNotFoundError:
    print(f"File not found: {input_path}")
except Exception as e:
    print(f"An error occurred: {e}")
