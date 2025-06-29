import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pycocotools.coco import COCO

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# -------------------------------
# Self-Attention Block (for the generator)
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
# Dataset Loader for FracAtlas segmentation
# -------------------------------
class FracAtlasDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask = (mask > 0.5).float()  # Binarize mask
        return image, mask


# -------------------------------
# U-Net Generator with Self-Attention in Bottleneck
# -------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.down1 = self.contract_block(in_channels, features, 4, 2, 1)  # 256->128
        self.down2 = self.contract_block(features, features * 2, 4, 2, 1)  # 128->64
        self.down3 = self.contract_block(features * 2, features * 4, 4, 2, 1)  # 64->32
        self.down4 = self.contract_block(features * 4, features * 8, 4, 2, 1)  # 32->16

        # Bottleneck
        self.bottleneck = self.contract_block(features * 8, features * 16, 4, 2, 1)  # 16->8
        self.attention = SelfAttention(features * 16)  # Self-attention block

        # Decoder
        self.up4 = self.expand_block(features * 16, features * 8, 4, 2, 1)  # 8->16
        self.up3 = self.expand_block(features * 16, features * 4, 4, 2, 1)  # 16->32
        self.up2 = self.expand_block(features * 8, features * 2, 4, 2, 1)  # 32->64
        self.up1 = self.expand_block(features * 4, features, 4, 2, 1)  # 64->128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
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
        bn = self.attention(bn)  # Apply self-attention
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
# PatchGAN Discriminator
# -------------------------------
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, features=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------
# Dice Loss Function
# -------------------------------
def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# -------------------------------
# Training Function for the GAN with Debug Logging
# -------------------------------
def train(generator, discriminator, dataloader, num_epochs, device, lr=2e-4, lambda_l1=100):
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        start_time = time.time()
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (imgs, masks) in enumerate(dataloader):
            print(f"  Processing batch {batch_idx + 1}/{len(dataloader)}")
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            fake_masks = generator(imgs)
            fake_input = torch.cat([imgs, fake_masks], dim=1)
            pred_fake = discriminator(fake_input)
            valid = torch.ones_like(pred_fake, device=device)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_masks, masks)
            loss_dice = dice_loss(fake_masks, masks)
            loss_G = loss_GAN + lambda_l1 * (loss_L1 + loss_dice)
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_input = torch.cat([imgs, masks], dim=1)
            pred_real = discriminator(real_input)
            loss_real = criterion_GAN(pred_real, valid)
            fake_input = torch.cat([imgs, fake_masks.detach()], dim=1)
            pred_fake = discriminator(fake_input)
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()

        avg_G_loss = epoch_G_loss / len(dataloader)
        avg_D_loss = epoch_D_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Time: {elapsed:.2f}s  Loss_G: {avg_G_loss:.4f}  Loss_D: {avg_D_loss:.4f}")

        # Save sample outputs every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_samples(generator, imgs, masks, epoch + 1, device)

    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")


def save_samples(generator, imgs, masks, epoch, device):
    generator.eval()
    with torch.no_grad():
        fake_masks = generator(imgs.to(device))
    # Create a grid showing original images, generated masks, and ground truth masks
    samples = torch.cat([imgs.cpu(), fake_masks.cpu(), masks.cpu()], dim=0)
    vutils.save_image(samples, f'samples_epoch_{epoch}.png', nrow=imgs.size(0), normalize=True)
    generator.train()


# -------------------------------
# Main Function
# -------------------------------
def main():
    # Set the paths based on your folder structure
    image_dir = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\FracAtlas\images\Fractured"
    mask_dir = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\FracAtlas\Utilities\Fracture Split"

    # Define transforms for images and masks
    transform_img = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    dataset = FracAtlasDataset(image_dir, mask_dir, transform_img, transform_mask)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize the models
    generator = UNetGenerator(in_channels=1, out_channels=1, features=64).to(device)
    discriminator = PatchGANDiscriminator(in_channels=2, features=64).to(device)

    num_epochs = 50
    learning_rate = 2e-4
    lambda_l1 = 100

    # Train the GAN segmentation model
    train(generator, discriminator, dataloader, num_epochs, device, lr=learning_rate, lambda_l1=lambda_l1)


if __name__ == '__main__':
    main()
