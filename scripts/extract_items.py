import cv2
from PIL import Image
import numpy as np
import os

PADDING = 80  # Padding around the bounding box
OUTPUT_RESOLUTION = (512, 512)  # Desired output resolution


output_img_idx = 0


def clear_dir(path):
    pass


def read_img(path):
    img = cv2.imread(path)
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect non-white areas
    _, thresh = cv2.threshold(bw_img, 250, 255, cv2.THRESH_BINARY_INV)

    # find bounding boxes
    connected_output = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    stats = connected_output[2]

    # filter out small components
    min_area = 7e4
    bboxes = [stat[:4] for stat in stats[1:] if stat[4] >= min_area]

    # process and save each bounding box
    for i in range(len(bboxes)):
        process_bboxes(img, bboxes[i])


def process_bboxes(
    image,
    bbox,
):
    global output_img_idx
    x, y, w, h = bbox
    cropped = image[y : y + h, x : x + w]
    padded_img = (
        np.ones(shape=(h + PADDING * 2, w + PADDING * 2, 3), dtype="uint8") * 255
    )
    padded_img[PADDING : PADDING + h, PADDING : PADDING + w] = cropped

    scale_h = OUTPUT_RESOLUTION[0] / padded_img.shape[0]
    scale_w = OUTPUT_RESOLUTION[1] / padded_img.shape[1]

    scale = min(scale_h, scale_w)  # get the resizing scale

    resized_img = cv2.resize(
        padded_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
    )

    output_path = f"output_imgs/{output_img_idx}.png"
    cv2.imwrite(output_path, resized_img)
    output_img_idx += 1


if __name__ == "__main__":
    read_img("src_imgs/2.png")
