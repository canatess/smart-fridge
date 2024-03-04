import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# Define the model architecture
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        # Adjust the dimensions of `skip` to match `x` before concatenation
        skip = F.interpolate(
            skip, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True
        )
        x = torch.cat([x, skip], dim=1)  # Use `dim` instead of `axis`
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


# Load the pre-trained model
model = build_unet()
checkpoint = torch.load("mold_model_comb.pth")
model.load_state_dict(checkpoint)
model.eval()


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = A.Compose(
    [
        A.Resize(224, 224),  # Resize images to 224x224
        A.Normalize(mean=mean, std=std),  # Normalize images using general mean and std
        ToTensorV2(),  # Convert image to PyTorch tensor
    ]
)


# Define the prediction function
def predict_image(image):
    # Apply transformations to the input image
    img_numpy = test_transform(image=np.array(image))["image"]
    image_tensor = img_numpy.unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Apply sigmoid activation and thresholding
    preds = torch.sigmoid(output)
    preds = output.detach().squeeze().numpy()

    preds = np.where(preds >= 0.5, 1, 0)
    pred_indices = np.where(preds == 1)

    masked_image = torch.squeeze(image_tensor).permute(1, 2, 0).numpy()

    masked_image = (masked_image * std) + mean

    # Visualize the predictions
    cmap = plt.get_cmap("jet")
    masked_image[pred_indices] = cmap(preds[pred_indices])[:, :3]

    # Normalize the image array to be between -1 and 1

    return masked_image


# Create Gradio interface
inputs = gr.Image(type="pil")  # Define input image shape
outputs = gr.Image(type="numpy")  # Define output image

app = gr.Interface(
    fn=predict_image,
    inputs=inputs,
    outputs=outputs,
    title="Image Segmentation",
    description="Segmentation of input image using a U-Net model.",
)

# Launch the app
app.launch()
