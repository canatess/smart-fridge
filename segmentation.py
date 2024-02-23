import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert image to PyTorch tensor
    ]
)


class CustomDataset(Dataset):
    CLASSES = ["Peach", "Mold", "Background"]
    CLASS_COLORS = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 0])]

    def __init__(self, root_path, images_paths, masks_paths, preprocessing=None):
        self.root_path = root_path
        self.images_fps = images_paths
        self.masks_fps = masks_paths

        self.images_fps = [
            os.path.join(root_path, "peach", image_id) for image_id in self.images_fps
        ]

        self.masks_fps = [
            os.path.join(root_path, "masks", image_id) for image_id in self.masks_fps
        ]

        self.classes = self.CLASSES
        self.class_colors = self.CLASS_COLORS[: len(self.classes)]
        self.preprocessing = preprocessing

    @staticmethod
    def get_nonblack_region(mask):
        mask_indices = np.argwhere(mask != 0)
        if len(mask_indices) == 0:
            return None
        (min_y, min_x), (max_y, max_x) = mask_indices.min(0), mask_indices.max(0) + 1
        return min_x, min_y, max_x, max_y

    def __getitem__(self, idx):
        # Read data
        image = cv2.imread(self.images_fps[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], cv2.IMREAD_GRAYSCALE)

        # Get bounding box around non-black regions in the mask
        bbox = self.get_nonblack_region(mask)

        if bbox:
            min_x, min_y, max_x, max_y = bbox
            # Crop the image and mask using the bounding box
            image = image[min_y:max_y, min_x:max_x]
            mask = mask[min_y:max_y, min_x:max_x]

        # Convert masks to one-hot encoded arrays
        masks = []
        for color in self.class_colors:
            # Create a mask for each class
            class_mask = (mask == color).astype(np.float32)
            masks.append(class_mask)

        # Stack masks along the channel axis
        mask = np.stack(masks, axis=-1)

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # Convert to PyTorch tensor
        image = ToTensorV2()(image=image)["image"]
        mask = ToTensorV2()(image=mask)["image"].permute(2, 0, 1)

        return image, mask

    def __len__(self):
        return len(self.images_fps)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_block, self).__init__()
        self.conv = conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_block(out_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class build_unet(nn.Module):
    def __init__(self):
        super(build_unet, self).__init__()
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = conv_block(512, 1024)

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        self.outputs = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)
        return outputs


def decode_segmentation(mask):
    # Get the index of the class with the highest probability for each pixel
    mask_binary = np.argmax(mask, axis=0)

    # Define colors for each class
    class_colors = [
        (255, 0, 0),  # Red for class 1
        (0, 255, 0),  # Green for class 2
        (0, 0, 0),  # Black for no class (background)
    ]

    seg_image = np.zeros(
        (mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8
    )

    # Assign colors corresponding to the argmax indexes
    for class_idx, color in enumerate(class_colors):
        seg_image[mask_binary == class_idx] = color

    return seg_image


def save_segmentation_results(image, preds_colored, output_dir):
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot original image and predicted segmentation
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("Original Image")
    ax[0].imshow(image / 255.0)

    ax[1].set_title("Predicted Segmentation")
    ax[1].imshow(preds_colored, cmap="gray")

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation_results.png"))
    plt.close()


def calculate_mold_ratio(image_path, fruit_name, output_dir):
    # Load the trained model based on the fruit name
    if fruit_name.lower() == "peach":
        model_path = "peach_model_bce.pth"  # Update with the actual path
    else:
        print("Fruit not supported!")
        return None

    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Apply transforms
    img_transformed = test_transform(img)

    # Load the model and perform segmentation
    model = build_unet()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    preds = torch.softmax(model(img_transformed.unsqueeze(0)), dim=1)

    # Convert tensors to numpy arrays
    img_transformed = (
        img_transformed.permute(1, 2, 0).detach().numpy()
    )  # Convert from (C, H, W) to (H, W, C)

    preds = preds.detach().squeeze().numpy()

    preds_colored = decode_segmentation(preds)
    # Save segmentation results
    save_segmentation_results(img_transformed, preds_colored, output_dir)

    return mold_ratio


# Example usage
image_path = "C:\\Users\\user\\Documents\\GitHub\\smart-fridge\\peach_spoiled.jpg"
fruit_name = "Peach"  # Replace with the fruit name
output_dir = "output"
mold_ratio = calculate_mold_ratio(image_path, fruit_name, output_dir)
print("Mold ratio:", mold_ratio)
