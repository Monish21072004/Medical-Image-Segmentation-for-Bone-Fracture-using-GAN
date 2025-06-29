import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os


def select_and_overlay_image():
    # Hide the main tkinter window
    root = Tk()
    root.withdraw()

    # Ask user to select an image file
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not image_path:
        print("No file selected.")
        return

    # Derive the mask path based on the selected image
    # You can adjust this logic if your mask folder structure differs
    base_name = os.path.basename(image_path)
    file_name_no_ext = os.path.splitext(base_name)[0]

    # Example mask path generation (adjust according to your folder structure)
    mask_folder = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\FracAtlas\Utilities\Fracture Split"
    mask_path = os.path.join(mask_folder, f"{file_name_no_ext}_mask.png")

    if not os.path.exists(mask_path):
        print(f"Mask not found for: {file_name_no_ext}")
        return

    # Load the original image (grayscale) and mask (grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert the original image to color so we can overlay the colored mask
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Apply a color map to the mask (this makes the mask visually distinctive)
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # Create an overlay by blending the original image with the mask
    overlay = cv2.addWeighted(image_color, 0.7, mask_color, 0.3, 0)

    # Display the results using matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask Image")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Call the function
select_and_overlay_image()
