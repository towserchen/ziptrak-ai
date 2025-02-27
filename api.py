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

# The number of concurrent is 1
IS_IDLE = True


@app.post("/detect")
async def upload_image(
    is_indoor: int = Form(1),
    save_processed_images: int = Form(0),
    file_key: str = Form(...),
):
    print(f'is_indoor: {is_indoor}')
    print(f'save_processed_images: {save_processed_images}')

    if is_indoor > 0:
        is_indoor = True
    else:
        is_indoor = False

    if save_processed_images > 0:
        save_processed_images = True
    else:
        save_processed_images = False
        
    # Read the file from S3

    if not detect_app.is_idle():
        raise HTTPException(status_code=400, detail="The detection process is not idle.")
    
    result = detect_app.detect_file(file_path, is_indoor, False, False)
    coordinate_list = []

    for coordinate in result:
        _coordinate = [[int(x) for x in point] for point in coordinate]
        coordinate_list.append(_coordinate)

    print(coordinate_list)

    return JSONResponse(content={
        "coordinate_list": coordinate_list
    })