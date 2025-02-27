from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

import time
import os

import app as detect_app
from middleware.APIVerification import APIVerification
from utils.helper import ajax
import boto3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(APIVerification)

# 5m
MAX_FILE_SIZE = 5 * 1024 * 1024

# The number of concurrent is 1
IS_IDLE = True


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    
    return ajax(0, f'{errors[0]['msg']}: {errors[0]['loc']}', None)


def download_file_from_s3(s3_bucket, s3_key):
    s3 = boto3.client("s3")

    ext = s3_key.split('.')[-1]
    local_file_path = f'./upload/{time.time()}.{ext}'
    s3.download_file(s3_bucket, s3_key, local_file_path)

    return local_file_path


@app.post('/node/is_available')
async def is_available():
    return ajax(detect_app.is_idle())


@app.post("/detect/image")
async def upload_image(
    is_indoor: int = Form(1),
    save_processed_images: int = Form(0),
    s3_bucket: str = Form(...),
    s3_key: str = Form(...),
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

    if not detect_app.is_idle():
        ajax(0, 'The detection process is not idle.', None)

    # Read the file from S3
    file_path = download_file_from_s3(s3_bucket, s3_key)
    
    try:
        result = detect_app.detect_file(file_path, is_indoor, save_processed_images, False)
    except Exception as e:
        ajax(0, f'Detect failed: {e}', None)

    coordinate_list = []

    for coordinate in result:
        _coordinate = [[int(x) for x in point] for point in coordinate]
        coordinate_list.append(_coordinate)

    print(coordinate_list)

    return ajax(1, 'ok', coordinate_list)