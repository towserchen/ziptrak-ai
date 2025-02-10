from PIL import ImageDraw, Image
import numpy as np
import torch
import cv2
import detect
from utils import tool
import sam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_file(file_path, is_window_detected=True, save_processed_images=True, show_final_image=True):
    """
    Detect the file

    Args:
        string file_path: The image path you want to detect, supports jpg, png
        bool is_window_detected: Detect thw windows or not. True for indoor detection, False for outdoor detection
        bool save_processed_images: Whether to save image that is detected in the state of object detection. The images will be stored at ./results/
        bool show_final_image: Whether to show the final image in a window, run locally only

    Returns:
        list openning coordinates
    """
    file_name = file_path.split('.')[0]

    if is_window_detected:
        score_thrushold = 0.15
    else:
        score_thrushold = 0.01

    # Detect
    detect_result = detect.inference_model(is_window_detected, file_path, "window", score_thrushold)

    label = [1, 1]
    masks = []
    scaled_masks = []
    cornersList = []

    if is_window_detected:
        # Filter out low score windows when multiple windows overlap
        detect_result = tool.filter_overlapped_windows(detect_result)
    else:
        detect_result = tool.filter_larger_rectangle(detect_result)

        # Filter out low score pillar when found more than 4 pillars
        if len(detect_result) > 4:
            detect_result = tool.filter_low_score_pillar_bboxes(detect_result)

    filtered_bboxes = [d['bbox'] for d in detect_result]

    image = Image.open(file_path)
    original_width, original_height = image.size

    index = 0

    for row in filtered_bboxes:
        index += 1

        new_row = np.array([[row[0], row[1]], [row[2], row[3]]])
        mask = sam.segment_with_boxs(image, new_row, label)

        mask = tool.filter_small_masks(mask)

        scaled_mask = cv2.resize(np.squeeze(mask, axis=0).astype(np.uint8),
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
        )

        masks.append(mask)
        scaled_masks.append(scaled_mask)


    # Filter out pillars that are too short
    if not is_window_detected:
        scaled_masks = tool.filter_short_masks(scaled_masks)
        scaled_masks = tool.sort_masks_left_to_right(scaled_masks)

        # Filter base on aspect ratio
        if len(scaled_masks) > 4:
            scaled_masks = tool.filter_masks_by_aspect_ratio_avg(scaled_masks)
        else:
            scaled_masks = tool.filter_masks_by_aspect_ratio(scaled_masks)

    print(len(scaled_masks))
    
    
    pillarCornerList = []

    for index, scaled_mask in enumerate(scaled_masks):
        # Corner sequence: left_top, right_top, right_bottom, left_bottom. clockwise
        corner = tool.get_corners(scaled_mask)
        cornersList.append([corner['top_left'], corner['top_right'], corner['bottom_right'], corner['bottom_left']])

        if not is_window_detected:
            if len(scaled_masks) <= 1:
                print('Only 1 pillar detected')
            else:
                # First pillar, fetch the rect's top_left, bottom_left as the coordinates of left_top and left_bottom
                if index == 0:
                    pillarCornerList.append([corner['top_left'], -1, -1, corner['bottom_left']])
                else:
                    # Not the first one, fetch rect's top_right, bottom_right as the previous pillar's right_top, right_bottom
                    pillarCornerList[index - 1][1] = corner['top_right']
                    pillarCornerList[index - 1][2] = corner['bottom_right']

                    # Not the last one, keep complete coordinates of pillars
                    if index != len(scaled_masks) - 1:
                        pillarCornerList.append([corner['top_right'], -1, -1, corner['bottom_right']])

    assert all(mask.shape == masks[0].shape for mask in masks), "All masks must have same shape"

    # Combine all masks
    combined_mask = np.sum(masks, axis=0)

    if save_processed_images:
        detect.save_result(file_name, combined_mask, image, device)

    if show_final_image:
        if not is_window_detected:
            tool.drawPolyLine(file_path, pillarCornerList)
        else:
            tool.drawPolyLine(file_path, cornersList)

    if is_window_detected:
        return cornersList
    else:
        return pillarCornerList


if __name__ == '__main__':
    # The image path you want to detect
    file_path = 'samples/16.jpg'

    # True for Indoor detection, otherwise False
    is_window_detected = False

    detect_file(file_path, is_window_detected, True, True)