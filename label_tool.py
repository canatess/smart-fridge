import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

# Initialize variables
# Replace with the path to the image folder
image_folder = "images"
annotations = {}  # Dictionary to store annotations for each image
drawing = False  # Flag for drawing bounding boxes
pt1 = (0, 0)
pt2 = (0, 0)
current_image_index = 0
class_label = ""

# Initialize Tkinter
root = tk.Tk()
root.title("Image Annotation Tool")

# Create a fixed-size Canvas to display the image
canvas_width = 500
canvas_height = 500
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# List of image file names in the specified folder
image_files = [
    f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Create an initial image variable
image = None
img = None


def center_image():
    global canvas, img
    canvas.delete("all")
    if img:
        # Calculate the position to center the image
        x = (canvas_width - img.width()) // 2  # Center the image horizontally
        y = (canvas_height - img.height()) // 2  # Center the image vertically

        canvas.create_image(x, y, image=img, anchor="nw", tags="image")


def load_image():
    global image, img
    if current_image_index < len(image_files):
        # If moving to the next image, save the annotations for the current image
        if current_image_index > 0:
            save_annotations()

        image_path = os.path.join(image_folder, image_files[current_image_index])
        image = Image.open(image_path)
        resized_image = image.resize((canvas_width, canvas_height))
        img = ImageTk.PhotoImage(resized_image)
        center_image()
        annotations = {}  # Clear annotations for the new image


load_image()


def draw_rect(event):
    global drawing, pt1, pt2, class_label, current_rect
    if drawing:
        pt2 = (event.x, event.y)
        canvas.coords(current_rect, pt1[0], pt1[1], pt2[0], pt2[1])


# Bind mouse events
def start_drawing(event):
    global drawing, pt1, pt2, class_label, current_rect
    if not drawing:
        pt1 = (event.x, event.y)
        pt2 = (event.x, event.y)
        current_rect = canvas.create_rectangle(
            pt1[0], pt1[1], pt2[0], pt2[1], outline="green", width=2, tags="current_box"
        )
        drawing = True


def stop_drawing(event):
    global drawing
    if drawing:
        drawing = False
        class_label = simpledialog.askstring(
            "Class Label",
            "Enter the class label for this object (or leave empty for NaN):",
        )
        if class_label is None:
            class_label = "NaN"
        x_center = (pt1[0] + pt2[0]) / 2.0
        y_center = (pt1[1] + pt2[1]) / 2.0
        width = abs(pt2[0] - pt1[0])
        height = abs(pt2[1] - pt1[1])
        annotations[current_image_index] = annotations.get(current_image_index, [])
        annotations[current_image_index].append(
            (class_label, x_center, y_center, width, height)
        )


def save_annotations():
    if not annotations:
        return

    annotation_file = image_files[current_image_index] + "_annotations.txt"
    with open(annotation_file, "w") as f:
        for annotation in annotations.get(current_image_index, []):
            class_label, x, y, width, height = annotation
            f.write(f"{class_label} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")


def load_next_image():
    global current_image_index, class_label
    current_image_index = (current_image_index + 1) % len(image_files)
    class_label = ""
    load_image()


canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<B1-Motion>", draw_rect)
canvas.bind("<ButtonRelease-1>", stop_drawing)

# Buttons with space between image and buttons
btn_save = tk.Button(root, text="Save Annotations to File", command=save_annotations)
btn_next = tk.Button(root, text="Next Image", command=load_next_image)

# Adjust the position of buttons
btn_save.place(x=10, y=canvas_height + 10)
btn_next.place(x=160, y=canvas_height + 10)

root.geometry(
    f"{canvas_width}x{canvas_height + 60}"
)  # Adjust window height to accommodate buttons
root.mainloop()
