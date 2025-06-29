import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils  # Added for RLE decoding


def coco_to_masks(coco_json_path, images_dir, output_mask_dir):
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    coco = COCO(coco_json_path)
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images in the annotations.")

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']

        # Initialize an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if 'segmentation' in ann:
                segm = ann['segmentation']

                if isinstance(segm, list):
                    # Polygon format
                    for poly in segm:
                        poly = np.array(poly).reshape((-1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], 255)

                elif isinstance(segm, dict):
                    # RLE format
                    rle = maskUtils.frPyObjects(segm, height, width)
                    decoded_mask = maskUtils.decode(rle) * 255
                    mask = np.maximum(mask, decoded_mask)

        mask_filename = os.path.splitext(file_name)[0] + '_mask.png'
        mask_save_path = os.path.join(output_mask_dir, mask_filename)
        cv2.imwrite(mask_save_path, mask)
        print(f"Saved mask: {mask_save_path}")


if __name__ == '__main__':
    coco_json_path = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\FracAtlas\Annotations\COCO JSON\COCO_fracture_masks.json"
    images_dir = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\FracAtlas\images\Fractured"
    output_mask_dir = r"C:\Users\Monish V\OneDrive\Documents\RANDOM_PROJECTS\Minor Project\masks"

    coco_to_masks(coco_json_path, images_dir, output_mask_dir)
