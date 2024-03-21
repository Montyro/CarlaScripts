from PIL import Image
import os
import shutil

# Define paths to the source folder and destination folder
source_folder = "D:/test/cam1"
destination_folder = "D:/test/"

# Create a non-valid folder inside the destination folder if it doesn't exist
non_valid_folder = os.path.join(destination_folder, "nonvalid")
if not os.path.exists(non_valid_folder):
    os.makedirs(non_valid_folder)
    ss_folder = os.path.join(non_valid_folder, "ss")
    rgb_folder = os.path.join(non_valid_folder, "rgb")
    os.makedirs(ss_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)


# Function to check if an image is valid
def is_image_valid(image_path):
    img = Image.open(image_path)
    width, height = img.size
    class_counts = {}
    #print(img.getdata())
    for pixel in img.getdata():
        if pixel in class_counts:
            class_counts[pixel] += 1
        else:
            class_counts[pixel] = 1
    
    #print(class_counts)
    for class_k, count in class_counts.items():
        if class_k[0] == 23 or class_k[0] == 10:
            if count / (width * height) > 0.4:
                return False

    return True

# Iterate through the ss folder
ss_folder = os.path.join(source_folder, "ss")
rgb_folder = os.path.join(source_folder, "rgb")

for filename in os.listdir(ss_folder):
    if filename.endswith(".png"):
        ss_image_path = os.path.join(ss_folder, filename)
        
        if not is_image_valid(ss_image_path):
            # Move the SS image to the non-valid folder
            destination_ss_path = os.path.join(non_valid_folder, "ss", filename)
            os.rename(ss_image_path, destination_ss_path)
            
            # Find and move the corresponding RGB image
            corresponding_rgb_path = os.path.join(rgb_folder, filename)
            if os.path.exists(corresponding_rgb_path):
                destination_rgb_path = os.path.join(non_valid_folder, "rgb", filename)
                os.rename(corresponding_rgb_path, destination_rgb_path)
                print(f"Moved {filename} to non-valid folder.")
        else:
            print(f"{filename} is valid.")

print("Processing completed.")
