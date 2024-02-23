import json
import os
import cv2
import numpy as np

# Define the directory where images are located
image_directory = "peach"  # Change this to your image directory

# Create a directory to store the mask images
os.makedirs("masks", exist_ok=True)

# Load and process each annotation file
annotation_files = ["annotations.json", "annotations_2.json"]

# Define pixel values for each class
class_labels = {"Mold": 255, "Peach": 128}

for file_name in annotation_files:
    with open(file_name, "r") as json_file:
        data = json.load(json_file)

    # Process each document
    for document in data:
        # Create an empty mask image of the same size as the original image
        image_path = os.path.join(image_directory, document["documents"][0]["name"])
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape

        # Initialize a combined mask image
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Process each entity in the annotation
        for entity in document["annotation"]["annotationGroups"][0][
            "annotationEntities"
        ]:
            entity_name = entity["name"]

            # Check if the entity is "Mold" or "Peach"
            if entity_name in class_labels:
                # Get the pixel value for the entity
                pixel_value = class_labels[entity_name]

                for block in entity["annotationBlocks"]:
                    for annotation in block["annotations"]:
                        segments = annotation["segments"]
                        # Convert the segments to int32 data type
                        polygons = [np.array(segment, np.int32) for segment in segments]

                        # Create a mask for the entity category
                        entity_mask = np.zeros(
                            (image_height, image_width), dtype=np.uint8
                        )
                        cv2.fillPoly(entity_mask, polygons, pixel_value)

                        # Add the entity mask to the combined mask
                        combined_mask = np.maximum(combined_mask, entity_mask)

        # Save the combined mask image
        image_filename = os.path.splitext(document["documents"][0]["name"])[0]
        mask_filename = f"masks/{image_filename}.jpg"
        cv2.imwrite(mask_filename, combined_mask)
