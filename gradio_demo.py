import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2


class RegressionModel(nn.Module):

    def __init__(self):

        super(RegressionModel, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # Remove the last fully connected layer
        self.regressor1 = nn.Linear(
            2048, 256
        )  # Replace the last layer with a regression layer
        self.regressor2 = nn.Linear(256, 1)

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor1(x)
        x = self.regressor2(x)
        return x


# Define the model
model_reg = RegressionModel()


model = torch.load("regression_model.pth", map_location="cpu")
model_reg.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def regression(image):

    img_numpy = test_transforms(Image.fromarray(image))
    image_tensor = img_numpy.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model_reg(image_tensor)
    print(output.shape)
    return output.squeeze().tolist()


def hugg_face(img):

    model = YOLO("plv7.pt")
    labels = [
        "freshpeach",
        "freshlemon",
        "rottenpeach",
        "rotten lemon",
        "freshmandarin",
        "rottenmandarin",
        "freshtomato",
        "rottentomato"
    ]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model(img)
    img_label_results = []
    img_2 = img.copy()

    for result in results:
        for i, cls in enumerate(result.boxes.cls):
            crop_img = img[
                int(result.boxes.xyxy[i][1]) : int(result.boxes.xyxy[i][3]),
                int(result.boxes.xyxy[i][0]) : int(result.boxes.xyxy[i][2]),
            ]
            cv2.rectangle(
                img_2,
                (int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1])),
                (int(result.boxes.xyxy[i][2]), int(result.boxes.xyxy[i][3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_2,
                labels[int(cls)] + str(i),
                (int(result.boxes.xyxy[i][0]), int(result.boxes.xyxy[i][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            img_label_results.append(
                {"label": labels[int(cls)] + str(i), "crop_img": crop_img}
            )

    img_2_pil = Image.fromarray(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
    regression_results = []

    # Iterate over cropped images and their labels
    for item in img_label_results:
        label = item["label"]
        cropped_img = item["crop_img"]
        # Apply regression to the cropped image
        regression_output = regression(cropped_img)
        # Append the regression output along with its label to the regression_results list
        regression_results.append(
            {"label": label, "Rotten Proportion": regression_output}
        )
    return img_2_pil, regression_results


# Create Gradio interface
inputs = gr.Image(type="pil")  # Define input image shape

# Define output types: one for the text box (for regression results) and one for the image (segmented image)
outputs = [
    gr.Image(type="pil", label="Detecion Result"),
    gr.Textbox(label="Regression Results"),
]

app = gr.Interface(
    fn=hugg_face,
    inputs=inputs,
    outputs=outputs,
    title="Smart Fridge with Regression",
    description="Rotten part regression results",
)

# Launch the app
app.launch(share=True)
