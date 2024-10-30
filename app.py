from PIL import ImageDraw, Image
import numpy as np
import torch
import cv2
import detect
from utils import tool
import sam

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import time
import os

# 5m
MAX_FILE_SIZE = 5 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()


def detect_file(file_path, is_window_detected=True, save_processed_images=True, show_final_image=True):

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
    
    
    pillarCornerList = []
    
    for index, scaled_mask in enumerate(scaled_masks):
        # Corner sequence: left_top, right_top, right_bottom, left_bottom. clockwise
        corner = tool.get_corners(scaled_mask)
        cornersList.append([corner['top_left'], corner['top_right'], corner['bottom_right'], corner['bottom_left']])

        if not is_window_detected:
            # First pillar, fetch the rect's right_top, right_bottom as the coordinates of left_top and left_bottom
            if index == 0:
                pillarCornerList.append([corner['top_right'], -1, -1, corner['bottom_right']])
            else:
                # Not the first one, fetch rect's left_top, left_bottom as the previous pillar's right_top, right_bottom
                pillarCornerList[index - 1][1] = corner['top_left']
                pillarCornerList[index - 1][2] = corner['bottom_left']

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
    file_path = 'samples/13.jpg'
    is_window_detected = False

    #detect_file(file_path, is_window_detected, True, True)



@app.post("/detect")
async def upload_image(
    upload_file: UploadFile = File(...),
    is_window_detected: int = 1,
    save_processed_images: int = 1
):    
    if is_window_detected > 0:
        is_window_detected = True
    else:
        is_window_detected = False

    if save_processed_images > 0:
        save_processed_images = True
    else:
        save_processed_images = False

    # 验证文件类型
    if upload_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
    
    # 验证文件大小
    if upload_file.file.seek(0, os.SEEK_END) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the limit of 5MB.")
    upload_file.file.seek(0)

    # 生成毫秒时间戳
    timestamp = int(time.time() * 1000)
    file_extension = os.path.splitext(upload_file.filename)[1]
    new_file_name = f"{timestamp}{file_extension}"

    # 处理上传的图片
    file_path = f"uploads/{new_file_name}"
    file_location = file_path
    with open(file_location, "wb") as file:
        file.write(await upload_file.read())

    result = detect_file(file_path, is_window_detected, False, False)
    coordinate_list = result[0]

    coordinate_list = [[int(x) for x in point] for point in coordinate_list]

    print(coordinate_list)

    # 返回 JSON 响应
    return JSONResponse(content={
        "coordinate_list": coordinate_list
    })