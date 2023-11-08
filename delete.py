"Delete images which is not annotated"


import os
import json

# Directory containing your images
image_directory = "peach"

# Load annotations from the first JSON file
with open("annotations.json", "r") as json_file:
    data = json.load(json_file)

# Load annotations from the second JSON file
with open("annotations_2.json", "r") as json_file:
    data_2 = json.load(json_file)

# Get a list of image filenames mentioned in both JSON files
image_filenames = set()
for document in data:
    image_filenames.add(document["documents"][0]["name"])

for document in data_2:
    image_filenames.add(document["documents"][0]["name"])

# Get a list of image filenames in the "peach" folder
all_images = os.listdir(image_directory)

# Find images not mentioned in both JSON files
images_to_delete = [image for image in all_images if image not in image_filenames]

# Delete the images that don't have a match
for image_to_delete in images_to_delete:
    image_path = os.path.join(image_directory, image_to_delete)
    os.remove(image_path)
    print(f"Deleted: {image_to_delete}")

print("Deletion process complete.")
