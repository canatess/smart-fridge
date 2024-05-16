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


# Define the Regression Model class
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Load pretrained ResNet101
        resnet = torchvision.models.resnet101(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Replace the last layer with regression layers
        self.regressor1 = nn.Linear(2048, 512)
        self.regressor2 = nn.Linear(512, 64)
        self.regressor3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor1(x)
        x = nn.GELU()(x)
        x = self.regressor2(x)
        x = nn.GELU()(x)
        x = self.regressor3(x)
        return x


# Load the pre-trained model state dictionary
model_state_dict = torch.load("regression_model.pth", map_location="cpu")
# Instantiate the RegressionModel
model_reg = RegressionModel()
# Load the state dictionary into the model
model_reg.load_state_dict(model_state_dict)
# Set the model to evaluation mode
model_reg.eval()

# Define transformations for test images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


# Define the regression function
def regression(image):
    img_numpy = test_transforms(Image.fromarray(image))
    image_tensor = img_numpy.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model_reg(image_tensor)
    return output.item()


# Define the object detection function
def hugg_face(img):
    # Load YOLO model
    model = YOLO("yolo.pt")
    labels = [
        "freshpeach",
        "freshlemon",
        "rottenpeach",
        "rotten lemon",
        "freshmandarin",
        "rottenmandarin",
        "freshtomato",
        "rottentomato",
        "freshcucumber",
        "rottencucumber",
    ]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = model(img)
    img_label_results = []
    img_2 = img.copy()

    # Process each detection result
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

    # Perform regression on each cropped image
    for item in img_label_results:
        label = item["label"]
        cropped_img = item["crop_img"]
        regression_output = regression(cropped_img)
        # Append regression results to the list
        regression_results.append(
            {"label": label, "Rotten Part Percentage": round(regression_output,2)}
        )
    return img_2_pil, regression_results


# Define Gradio interface
inputs = gr.Image(type="pil")
outputs = [
    gr.Image(type="pil", label="Detection Result"),  # Output for the segmented image
    gr.Textbox(label="Regression Results"),  # Output for the regression results
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
