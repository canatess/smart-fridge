import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Initialize variables
# Replace with the path to your image folder
image_folder = 'images'
annotations = {}  # Dictionary to store annotations for each image
drawing = False  # Flag for drawing bounding boxes
pt1 = (0, 0)
pt2 = (0, 0)
current_image_index = 0
class_label = ""

# Initialize Tkinter
root = tk.Tk()
root.title('Image Annotation Tool')

# Create a Canvas to display the image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# List of image file names in the specified folder
image_files = [f for f in os.listdir(
    image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Create an initial image variable
image = None
img = None


def center_image():
    global canvas, img  # Make 'canvas' and 'img' global variables
    canvas.delete("all")
    if img:
        # Get the image dimensions
        img_width = img.width()
        img_height = img.height()

        # Calculate the canvas size based on the image size
        canvas_width = min(800, img_width)
        canvas_height = min(600, img_height)

        # Center the image
        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2

        canvas.config(width=canvas_width, height=canvas_height)
        canvas.create_image(x, y, image=img, anchor=tk.NW)


def load_image():
    global image, img, annotations
    if current_image_index < len(image_files):
        # If moving to the next image, save the annotations for the current image
        if current_image_index > 0:
            save_annotations()

        image_path = os.path.join(
            image_folder, image_files[current_image_index])
        image = Image.open(image_path)
        img = ImageTk.PhotoImage(image)
        center_image()
        annotations = {}  # Clear annotations for the new image


load_image()


def draw_rect(event):
    global drawing, pt1, pt2, class_label
    if drawing:
        # Clear the canvas
        canvas.delete("current_box")

        pt2 = (event.x, event.y)
        canvas.create_rectangle(
            pt1[0], pt1[1], pt2[0], pt2[1], outline='green', width=2, tags="current_box")
        class_label = simpledialog.askstring(
            "Class Label", "Enter the class label for this object (or leave empty for NaN):")
        if class_label is None:
            class_label = "NaN"
        # Save the annotation as (class_label, x, y, width, height)
        x_center = (pt1[0] + pt2[0]) / 2.0
        y_center = (pt1[1] + pt2[1]) / 2.0
        width = abs(pt2[0] - pt1[0])
        height = abs(pt2[1] - pt1[1])

        # Add the annotation to the current image's annotations
        if current_image_index not in annotations:
            annotations[current_image_index] = []
        annotations[current_image_index].append(
            (class_label, x_center, y_center, width, height))

        drawing = False
    else:
        pt1 = (event.x, event.y)
        pt2 = (event.x, event.y)
        drawing = True


def save_annotations():
    if not annotations:
        return

    annotation_file = image_files[current_image_index] + '_annotations.txt'
    with open(annotation_file, 'w') as f:
        for annotation in annotations.get(current_image_index, []):
            class_label, x, y, width, height = annotation
            f.write(f"{class_label} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")


def load_next_image():
    global current_image_index, class_label
    current_image_index = (current_image_index + 1) % len(image_files)
    class_label = ""
    load_image()


canvas.bind("<Button-1>", draw_rect)

# Anchor buttons to the bottom-right corner
btn_save = tk.Button(root, text="Save Annotations to File",
                     command=save_annotations)
btn_next = tk.Button(root, text="Next Image", command=load_next_image)

btn_save.pack(side=tk.RIGHT)
btn_next.pack(side=tk.RIGHT)

root.mainloop()
