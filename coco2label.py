"""
Create masks of mold class for now from toras formatted annotations
"""

import json
import os
import cv2
import numpy as np

# Load the first JSON data
with open("annotations.json", "r") as json_file:
    data = json.load(json_file)

# Define the directory where images are located
image_directory = "peach"  # Change this to your image directory

# Create a directory to store the mask images
os.makedirs("masks", exist_ok=True)

# Process each document in the first data file
for document in data:
    # Create an empty mask image of the same size as the original image
    image_path = os.path.join(image_directory, document["documents"][0]["name"])
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Initialize a combined mask image for Mold category
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for entity in document["annotation"]["annotationGroups"][0]["annotationEntities"]:
        entity_name = entity["name"]

        # Check if the entity is "Mold"
        if entity_name == "Mold":
            for block in entity["annotationBlocks"]:
                for annotation in block["annotations"]:
                    segments = annotation["segments"]
                    # Convert the segments to int32 data type
                    polygons = [np.array(segment, np.int32) for segment in segments]

                    # Create a mask for the "Mold" category
                    mold_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    cv2.fillPoly(
                        mold_mask, polygons, 255
                    )  # Assign white color for "Mold"

                    # Add the "Mold" mask to the combined mask
                    combined_mask = cv2.add(combined_mask, mold_mask)

    # Save the combined "Mold" mask image ( For now not using this combined but when we need other classes suc has peach it will be useful)
    image_filename = os.path.splitext(document["documents"][0]["name"])[0]
    mask_filename = f"masks/{image_filename}.png"
    cv2.imwrite(mask_filename, combined_mask)

# Load the second JSON data (annotations from annotations_2.json)
with open("annotations_2.json", "r") as json_file:
    data_2 = json.load(json_file)

# Process each document in the second data file
for document in data_2:
    # Create an empty mask image of the same size as the original image
    image_path = os.path.join(image_directory, document["documents"][0]["name"])
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Initialize a combined mask image
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for entity in document["annotation"]["annotationGroups"][0]["annotationEntities"]:
        entity_name = entity["name"]

        # Check if the entity is the category you want
        if entity_name == "Mold":
            for block in entity["annotationBlocks"]:
                for annotation in block["annotations"]:
                    segments = annotation["segments"]
                    # Convert the segments to int32 data type
                    polygons = [np.array(segment, np.int32) for segment in segments]

                    mold_mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    cv2.fillPoly(
                        mold_mask, polygons, 255
                    )  # Assign white color for "NewCategory"

                    combined_mask = cv2.add(combined_mask, mold_mask)

    # Save the combined mask image
    image_filename = os.path.splitext(document["documents"][0]["name"])[0]
    mask_filename = f"masks/{image_filename}.png"
    cv2.imwrite(mask_filename, combined_mask)
