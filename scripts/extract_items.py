import cv2
import numpy as np
import os

INPUT_FOLDER = "src_imgs"  # Folder containing input images

PADDING = 80  # Padding around the bounding box
OUTPUT_RESOLUTION = (512, 512)  # Desired output resolution
OUTPUT_FOLDER = "output_imgs"

output_img_idx = 0


def clear_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def ensure_output_dir():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print("Output directory created:", OUTPUT_FOLDER)
    else:
        print("Output directory found:", OUTPUT_FOLDER)
        clear_dir(OUTPUT_FOLDER)


def process_img(path):
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

    # pad and scale the image to match the output resolution
    x, y, w, h = bbox
    cropped = image[y : y + h, x : x + w]
    padded_img = np.full((h + PADDING * 2, w + PADDING * 2, 3), 255, dtype="uint8")
    padded_img[PADDING : PADDING + h, PADDING : PADDING + w] = cropped

    scale_h = OUTPUT_RESOLUTION[0] / padded_img.shape[0]
    scale_w = OUTPUT_RESOLUTION[1] / padded_img.shape[1]

    scale = min(scale_h, scale_w)  # get the resizing scale

    resized_img = cv2.resize(
        padded_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
    )

    # fill in the missing pixels to match the output resolution
    output_img = np.full(
        (OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1], 3), 255, dtype=np.uint8
    )
    h_offset = (OUTPUT_RESOLUTION[0] - resized_img.shape[0]) // 2
    w_offset = (OUTPUT_RESOLUTION[1] - resized_img.shape[1]) // 2
    output_img[
        h_offset : h_offset + resized_img.shape[0],
        w_offset : w_offset + resized_img.shape[1],
    ] = resized_img

    output_path = f"output_imgs/{output_img_idx}.png"
    cv2.imwrite(output_path, output_img)
    output_img_idx += 1


def extract_items():
    ensure_output_dir()
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".png"):
            img_path = os.path.join(INPUT_FOLDER, filename)
            process_img(img_path)


if __name__ == "__main__":
    extract_items()
    print(f"Extracted {output_img_idx} items to {OUTPUT_FOLDER}")
