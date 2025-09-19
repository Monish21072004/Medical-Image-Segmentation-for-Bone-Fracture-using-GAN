# 🩻 Medical Image Segmentation for Bone Fracture using GAN

A deep learning project that uses **Generative Adversarial Networks (GANs)** to segment **bone fractures** from X-ray images.
The system automatically detects and highlights fracture regions by generating segmentation masks, making it useful for **medical research, diagnostics, and educational purposes**.

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Usage](#usage)
7. [How It Works](#how-it-works)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

---

## 🌟 Overview

Detecting bone fractures manually can be time-consuming and prone to errors.
This project leverages **GANs** to **automatically segment fractures** from X-ray images, generating precise **binary masks** that highlight fracture areas.

**Why GANs?**

* GANs are powerful for **image-to-image tasks**, such as segmentation.
* The **Generator** predicts masks, while the **Discriminator** ensures masks are realistic and accurate.

---

## 📂 Dataset

We use the **[FracAtlas Dataset](https://www.kaggle.com/datasets/abdohamdg/fracatlas-dataset)**:

* **Description**: A curated dataset of X-ray images with annotated fracture regions.
* **Contents**:

  * Original X-ray images.
  * Segmentation masks marking fractured areas.
* **Usage**:

  * Train the GAN model.
  * Validate segmentation performance.
  * Test real-world applicability.

> ⚠️ *Note*: Ensure you comply with Kaggle’s dataset license before using it in production or redistribution.

**Folder structure:**

```
data/
├── images/      # X-ray images
├── masks/       # Ground truth fracture masks
└── splits/      # Train, validation, and test sets
```

---

## ✨ Features

* **Automated fracture detection** via segmentation masks.
* **GUI** for visualizing predictions interactively.
* Support for **training**, **validation**, and **testing** workflows.
* Easy to integrate with other medical imaging pipelines.
* Modular structure for experimentation with different GAN architectures.

---

## 🗂️ Project Structure

```
Medical-Image-Segmentation-for-Bone-Fracture-using-GAN/
├── bone.py            # Main segmentation pipeline
├── mask.py            # Mask generation utilities
├── trymask.py         # Experimental mask generation methods
├── gui.py             # GUI for viewing results interactively
├── models/            # (Optional) Pre-trained GAN weights
├── data/              # Dataset folder (FracAtlas images + masks)
├── sample_results/    # Example outputs (predicted masks)
└── README.md          # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Monish21072004/Medical-Image-Segmentation-for-Bone-Fracture-using-GAN.git
cd Medical-Image-Segmentation-for-Bone-Fracture-using-GAN
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` with the following common dependencies if not already present:

```
numpy
opencv-python
matplotlib
torch
torchvision
pillow
tkinter
```

Then install them:

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the [FracAtlas Dataset](https://www.kaggle.com/datasets/abdohamdg/fracatlas-dataset) and place it inside the `data/` directory:

```
data/
├── images/
└── masks/
```

---

## 🚀 Usage

| Script       | Purpose                                      |
| ------------ | -------------------------------------------- |
| `bone.py`    | Run segmentation on a given image or folder. |
| `mask.py`    | Generate or process segmentation masks.      |
| `trymask.py` | Experimental mask processing workflows.      |
| `gui.py`     | Launch GUI for interactive visualization.    |

### Example: Run Segmentation

```bash
python bone.py --input data/images/sample1.png --output sample_results/sample1_mask.png
```

### Example: Launch GUI

```bash
python gui.py
```

---

## 🧠 How It Works

The system uses a **GAN-based architecture** for medical image segmentation:

1. **Generator**:

   * Takes an X-ray image as input.
   * Outputs a binary segmentation mask of fracture areas.

2. **Discriminator**:

   * Distinguishes between real (ground truth) masks and generated masks.
   * Improves Generator performance through adversarial feedback.

3. **Training**:

   * Images and ground-truth masks from FracAtlas are used.
   * Loss = Adversarial Loss + Segmentation Loss (e.g., Dice or BCE loss).

---

## 📊 Evaluation Metrics

To measure segmentation quality, we recommend:

| Metric                  | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| **IoU (Jaccard Index)** | Measures overlap between predicted and ground truth masks. |
| **Dice Coefficient**    | Similarity measure between sets of pixels.                 |
| **Precision & Recall**  | Balance between false positives and false negatives.       |
| **Visual Inspection**   | Overlay masks on original images to check accuracy.        |

---

## 🔮 Future Improvements

* Add **pre-trained model weights** for quick testing.
* Implement a **training script** for full GAN training.
* Enhance GUI for better interactivity and reporting.
* Include more evaluation metrics and automatic reporting.
* Optimize performance for real-time segmentation.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repo.
2. **Create** a feature branch.
3. **Commit** your changes.
4. **Push** the branch and open a **Pull Request**.

---

## 📜 License

This project is released under the **MIT License**.
Please review the FracAtlas dataset license before using it commercially.

---

## 📧 Contact

**Author**: Monish V

* GitHub: [Monish21072004](https://github.com/Monish21072004)
* Email: [monishv217@gmail.com](mailto:monishv217@gmail.com)

---

## 🖼 Sample Output (Add Example Images Here)

| Original X-ray                     | Predicted Mask                   | Overlay                                |
| ---------------------------------- | -------------------------------- | -------------------------------------- |
| ![Input](sample_results/input.png) | ![Mask](sample_results/mask.png) | ![Overlay](sample_results/overlay.png) |

---

## 🌐 References

* [FracAtlas Dataset](https://www.kaggle.com/datasets/abdohamdg/fracatlas-dataset)
* GANs for Image Segmentation — Goodfellow et al., 2014
* PyTorch Documentation — [https://pytorch.org/docs/](https://pytorch.org/docs/)


