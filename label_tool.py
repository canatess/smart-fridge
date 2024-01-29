import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from itertools import chain

# Initialize variables
image_folder = ""
annotations = {}  # Dictionary to store annotations for each image
drawing = False  # Flag for drawing bounding boxes
pt1 = (0, 0)
pt2 = (0, 0)
current_image_index = 0
class_labels = []
polygon_drawing = False  # Flag for drawing polygons
polygon_points = []  # List to store points for the current polygon
current_polygon = None  # Variable to keep track of the current polygon
canvas = None  # Canvas widget
polygon_start_threshold = 8  # Distance threshold for closing the polygon

# List of image file names in the specified folder
image_files = []
annotation_mode = "polygon"


# Function to add a new class to the listbox
def add_class():
    global class_labels
    new_class = class_entry.get().strip()
    if new_class and new_class not in class_labels:
        class_labels.append(new_class)
        class_listbox.insert(tk.END, new_class)
        class_entry.delete(0, tk.END)  # Clear the text in the entry widget


# Function to select the folder and configure the start_button
def select_folder():
    global image_folder, image_files, canvas
    image_folder = filedialog.askdirectory()
    if image_folder:
        folder_label.config(text=f"Selected Folder: {image_folder}")
        image_files = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        canvas = tk.Canvas(welcome_root, width=500, height=500)
        canvas.pack()  # Create the canvas here
        start_button.config(command=load_image)  # Set the command to load_image
        welcome_root.update()  # Update the window to display the folder path


def select_annotation_mode(mode):
    global annotation_mode
    annotation_mode = mode
    start_button.config(command=start_labeling)
    welcome_root.update()


def start_labeling():
    welcome_root.destroy()


# Initialize Tkinter for welcome page
welcome_root = tk.Tk()
welcome_root.title("Welcome to Image Annotation Tool")

# Label to display the selected folder path
folder_label = tk.Label(welcome_root, text="Selected Folder: ")
folder_label.pack(pady=5)

# Frame for mode selection
mode_selection_frame = tk.Frame(welcome_root)
mode_selection_frame.pack(pady=10)

# Radio buttons to choose annotation mode
bbox_radio = tk.Radiobutton(
    mode_selection_frame,
    text="Bounding Box (bbox)",
    variable=annotation_mode,
    value="bbox",
    command=lambda: select_annotation_mode("bbox"),
)
bbox_radio.grid(row=0, column=0, padx=10)

polygon_radio = tk.Radiobutton(
    mode_selection_frame,
    text="Polygon",
    variable=annotation_mode,
    value="polygon",
    command=lambda: select_annotation_mode("polygon"),
)
polygon_radio.grid(row=0, column=1, padx=10)

# Button to select image folder
select_folder_button = tk.Button(
    welcome_root, text="Select Image Folder", command=select_folder
)
select_folder_button.pack(pady=10)

# Listbox for class names
class_listbox = tk.Listbox(
    welcome_root, selectmode=tk.MULTIPLE, exportselection=0, width=40, height=5
)
class_listbox.pack(pady=10)

# Entry for adding new class
class_entry = tk.Entry(welcome_root, width=30)
class_entry.pack(pady=5)

# Button to add class
add_class_button = tk.Button(welcome_root, text="Add Class", command=add_class)
add_class_button.pack(pady=5)

# Button to start annotation
start_button = tk.Button(welcome_root, text="Start Annotation", command=start_labeling)
start_button.pack(pady=10)

welcome_root.mainloop()


# Initialize Tkinter for main annotation window
root = tk.Tk()
root.title("Image Annotation Tool")

# Create a fixed-size Canvas to display the image
canvas_width = 500
canvas_height = 500
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()


def center_image():
    global canvas, img
    canvas.delete("all")
    if img:
        x = (canvas_width - img.width()) // 2
        y = (canvas_height - img.height()) // 2
        canvas.create_image(x, y, image=img, anchor="nw", tags="image")


def load_image():
    global image, img, current_image_index, annotations, polygon_points, current_polygon
    if current_image_index < len(image_files):
        if current_image_index > 0:
            save_annotations()

        # Clear existing annotations and related variables
        annotations = {}
        polygon_points = []
        if current_polygon:
            canvas.delete(current_polygon)

        image_path = os.path.join(image_folder, image_files[current_image_index])
        image = Image.open(image_path)
        resized_image = image.resize((canvas_width, canvas_height))
        img = ImageTk.PhotoImage(resized_image)
        center_image()


load_image()


def draw_rect(event):
    global drawing, pt1, pt2, class_label, current_rect
    if drawing:
        pt2 = (event.x, event.y)
        canvas.coords(current_rect, pt1[0], pt1[1], pt2[0], pt2[1])


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
    global drawing, class_labels
    if drawing:
        drawing = False

        # Ensure class_labels is not empty before creating the popup
        if class_labels:
            # Create a new window to ask for the class label
            class_label_popup = tk.Toplevel(root)
            class_label_popup.title("Class Label")

            # Label and Combobox
            label = tk.Label(class_label_popup, text="Select Class Label:")
            label.grid(row=0, column=0, padx=10, pady=10)

            class_label_combobox = ttk.Combobox(
                class_label_popup, values=class_labels, state="readonly"
            )

            class_label_combobox.grid(row=0, column=1, padx=10, pady=10)

            def save_annotation():
                selected_class_index = (
                    class_label_combobox.current()
                )  # Get the index of the selected class
                if selected_class_index != -1:  # Check if a class is selected
                    x_center = (pt1[0] + pt2[0]) / 2.0
                    y_center = (pt1[1] + pt2[1]) / 2.0
                    width = abs(pt2[0] - pt1[0])
                    height = abs(pt2[1] - pt1[1])

                    annotations[current_image_index] = annotations.get(
                        current_image_index, []
                    )
                    annotations[current_image_index].append(
                        (selected_class_index, x_center, y_center, width, height)
                    )

                    # Optionally, you can close the popup or perform other actions here

                    # Destroy the popup window
                    class_label_popup.destroy()

            # Button to save annotation
            save_button = tk.Button(
                class_label_popup, text="Save Annotation", command=save_annotation
            )
            save_button.grid(row=1, column=0, columnspan=2, pady=10)

        else:
            # Provide feedback that class_labels is empty
            print("Error: class_labels is empty")


def draw_polygon(event):
    global polygon_drawing, polygon_points, current_polygon
    if polygon_drawing:
        x, y = event.x, event.y
        canvas.coords(current_polygon, list(chain(*polygon_points, [x, y])))


def start_drawing_polygon(event):
    global polygon_drawing, polygon_points, current_polygon
    if not polygon_drawing:
        x, y = event.x, event.y
        polygon_points = [(x, y)]
        current_polygon = canvas.create_line(
            x, y, x, y, fill="green", width=2, smooth=tk.FALSE, tags="current_polygon"
        )
        # Draw a filled circle at the clicked point to make it more visible
        canvas.create_oval(
            x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", tags="current_point"
        )
        polygon_drawing = True


def stop_drawing_polygon(event):
    global polygon_drawing, class_labels, current_polygon
    if polygon_drawing:
        x, y = event.x, event.y
        distance_to_start = (
            (x - polygon_points[0][0]) ** 2 + (y - polygon_points[0][1]) ** 2
        ) ** 0.5
        if distance_to_start < polygon_start_threshold and len(polygon_points) > 2:
            polygon_drawing = False

            # Draw a line from the last point to the starting point
            canvas.create_line(
                polygon_points[-1][0],
                polygon_points[-1][1],
                polygon_points[0][0],
                polygon_points[0][1],
                fill="green",
                width=2,
                smooth=tk.FALSE,
                tags="current_polygon",
            )

            # Draw filled circles at each point in the polygon
            for point in polygon_points:
                x, y = point
                canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", tags="points"
                )

            # Ensure class_labels is not empty before creating the popup
            if class_labels:
                # Create a new window to ask for the class label
                class_label_popup = tk.Toplevel(root)
                class_label_popup.title("Class Label")

                # Label and Combobox
                label = tk.Label(class_label_popup, text="Select Class Label:")
                label.grid(row=0, column=0, padx=10, pady=10)

                class_label_combobox = ttk.Combobox(
                    class_label_popup, values=class_labels, state="readonly"
                )

                class_label_combobox.grid(row=0, column=1, padx=10, pady=10)

                def save_annotation():
                    selected_class_index = class_label_combobox.current()
                    if selected_class_index != -1:
                        annotations[current_image_index] = annotations.get(
                            current_image_index, []
                        )
                        annotations[current_image_index].append(
                            (selected_class_index, polygon_points.copy())
                        )

                        # Change the color of the completed polygon
                        canvas.itemconfig(
                            current_polygon, fill="blue"
                        )  # You can use any color

                        class_label_popup.destroy()

                # Button to save annotation
                save_button = tk.Button(
                    class_label_popup, text="Save Annotation", command=save_annotation
                )
                save_button.grid(row=1, column=0, columnspan=2, pady=10)

            else:
                print("Error: class_labels is empty")
        else:
            # Add the current point to the list of polygon points
            polygon_points.append((x, y))
            # Draw a line from the previous point to the current point
            canvas.create_line(
                polygon_points[-2][0],
                polygon_points[-2][1],
                x,
                y,
                fill="green",
                width=2,
                smooth=tk.FALSE,
                tags="current_polygon",
            )
            # Draw a filled circle at the current point
            canvas.create_oval(
                x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", tags="points"
            )


def save_annotations():
    if not annotations:
        return

    annotation_file = image_files[current_image_index] + "_annotations.txt"
    with open(annotation_file, "w") as f:
        for annotation in annotations.get(current_image_index, []):
            if annotation_mode == "bbox":
                class_label, x, y, width, height = annotation
                f.write(f"{class_label} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")
            elif annotation_mode == "polygon":
                class_label, points = annotation
                f.write(f"{class_label} {' '.join(map(str, chain(*points)))}\n")


def load_next_image():
    global current_image_index, class_label, annotations, polygon_points, current_polygon
    current_image_index = (current_image_index + 1) % len(image_files)
    class_label = ""

    # Clear existing annotations and related variables
    annotations = {}
    polygon_points = []
    if current_polygon:
        canvas.delete(current_polygon)

    load_image()


if annotation_mode == "bbox":
    canvas.bind("<ButtonPress-1>", start_drawing)
    canvas.bind("<B1-Motion>", draw_rect)
    canvas.bind("<ButtonRelease-1>", stop_drawing)

elif annotation_mode == "polygon":
    canvas.bind("<ButtonPress-1>", start_drawing_polygon)
    canvas.bind("<B1-Motion>", draw_polygon)
    canvas.bind("<ButtonRelease-1>", stop_drawing_polygon)


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
