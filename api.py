from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import time
import os

import app as detect_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5m
MAX_FILE_SIZE = 5 * 1024 * 1024


@app.post("/detect")
async def upload_image(
    upload_file: UploadFile = File(...),
    is_window_detected: int = Form(1),
    save_processed_images: int = Form(1)
):
    print(is_window_detected)
    print(save_processed_images)

    if is_window_detected > 0:
        is_window_detected = True
    else:
        is_window_detected = False

    if save_processed_images > 0:
        save_processed_images = True
    else:
        save_processed_images = False

    print(is_window_detected)

    if upload_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG and PNG are allowed.")
    
    if upload_file.file.seek(0, os.SEEK_END) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the limit of 5MB.")
    upload_file.file.seek(0)

    timestamp = int(time.time() * 1000)
    file_extension = os.path.splitext(upload_file.filename)[1]
    new_file_name = f"{timestamp}{file_extension}"

    file_path = f"uploads/{new_file_name}"
    file_location = file_path
    with open(file_location, "wb") as file:
        file.write(await upload_file.read())

    result = detect_app.detect_file(file_path, is_window_detected, False, False)
    coordinate_list = []

    print(result)

    for coordinate in result:
        _coordinate = [[int(x) for x in point] for point in coordinate]
        coordinate_list.append(_coordinate)

    print(coordinate_list)

    return JSONResponse(content={
        "coordinate_list": coordinate_list
    })