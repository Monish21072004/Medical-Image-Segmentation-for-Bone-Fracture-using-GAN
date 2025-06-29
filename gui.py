import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import cv2


# -------------------------------
# Define Self-Attention Block
# -------------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


# -------------------------------
# Define U-Net Generator with Self-Attention
# -------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.down1 = self.contract_block(in_channels, features, 4, 2, 1)  # 256 -> 128
        self.down2 = self.contract_block(features, features * 2, 4, 2, 1)  # 128 -> 64
        self.down3 = self.contract_block(features * 2, features * 4, 4, 2, 1)  # 64 -> 32
        self.down4 = self.contract_block(features * 4, features * 8, 4, 2, 1)  # 32 -> 16

        # Bottleneck
        self.bottleneck = self.contract_block(features * 8, features * 16, 4, 2, 1)  # 16 -> 8
        self.attention = SelfAttention(features * 16)  # Apply self-attention

        # Decoder
        self.up4 = self.expand_block(features * 16, features * 8, 4, 2, 1)  # 8 -> 16
        self.up3 = self.expand_block(features * 16, features * 4, 4, 2, 1)  # 16 -> 32
        self.up2 = self.expand_block(features * 8, features * 2, 4, 2, 1)  # 32 -> 64
        self.up1 = self.expand_block(features * 4, features, 4, 2, 1)  # 64 -> 128

        # Final layer: using Sigmoid so output is in [0, 1]
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def contract_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def expand_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bn = self.bottleneck(d4)
        bn = self.attention(bn)
        up4 = self.up4(bn)
        up4 = torch.cat([up4, d4], dim=1)
        up3 = self.up3(up4)
        up3 = torch.cat([up3, d3], dim=1)
        up2 = self.up2(up3)
        up2 = torch.cat([up2, d2], dim=1)
        up1 = self.up1(up2)
        up1 = torch.cat([up1, d1], dim=1)
        out = self.final(up1)
        return out


# -------------------------------
# Function to Load the Pretrained Generator
# -------------------------------
def load_generator(model_path, device):
    generator = UNetGenerator(in_channels=1, out_channels=1, features=64)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator


# -------------------------------
# Inference Function
# -------------------------------
def run_inference(generator, image_path, device):
    # Use same transforms as training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = generator(image_tensor)
    # Since output is already in [0, 1], threshold if a binary mask is desired:
    pred_mask = (pred_mask > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
    return image, pred_mask


# -------------------------------
# Function to Display Results using Matplotlib
# -------------------------------
def display_results(image, mask):
    image_np = np.array(image)
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    # Create an overlay
    image_color = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    if image_color.shape != mask_color.shape:
        mask_color = cv2.resize(mask_color, (image_color.shape[1], image_color.shape[0]))
    overlay = cv2.addWeighted(image_color, 0.7, mask_color, 0.3, 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_np, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_resized, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# -------------------------------
# Tkinter GUI for Inference
# -------------------------------
def select_and_run(generator, device):
    file_path = filedialog.askopenfilename(
        title="Select Image for Segmentation",
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )
    if file_path:
        image, pred_mask = run_inference(generator, file_path, device)
        display_results(image, pred_mask)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load pretrained generator
    generator_path = "generator_final.pth"  # Make sure this file exists in your working directory
    generator = load_generator(generator_path, device)

    # Create Tkinter window
    root = tk.Tk()
    root.title("Bone Fracture Segmentation")
    root.geometry("400x200")

    btn = tk.Button(root, text="Select Image for Segmentation",
                    command=lambda: select_and_run(generator, device),
                    font=("Helvetica", 12), padx=10, pady=10)
    btn.pack(expand=True)
    root.mainloop()


if __name__ == '__main__':
    main()
