from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

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


def get_form_data(request: Request):
    form_field_list = getattr(request.state, "form_field_list", {})
    file_list = getattr(request.state, "file_list", {})
    return form_field_list, file_list


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
    pid = os.getpid()
    print(f'Available: PID: {pid}')
    return ajax(1, 'ok', detect_app.is_idle())


@app.post("/detect/image")
async def upload_image(
    request: Request,
    form_data: tuple = Depends(get_form_data)
):
    pid = os.getpid()
    print(f'Detect: PID: {pid}')

    form_field_list, file_list = form_data

    is_indoor = int(form_field_list.get('is_indoor', 1))
    save_processed_images = int(form_field_list.get('save_processed_images', 0))
    s3_bucket = form_field_list.get('s3_bucket', None)
    s3_key = form_field_list.get('s3_key', None)
    upload_file = file_list.get('upload_file', None)

    print(f'is_indoor: {is_indoor}, save_processed_images: {save_processed_images}')

    if is_indoor > 0:
        is_indoor = True
    else:
        is_indoor = False

    if save_processed_images > 0:
        save_processed_images = True
    else:
        save_processed_images = False

    if not detect_app.is_idle():
        return ajax(0, 'The detection process is not idle.', None)

    if upload_file:
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
    else:
        print('download from s3')
        # Read the file from S3
        file_path = download_file_from_s3(s3_bucket, s3_key)

    print(f'file_path: {file_path}')
    
    try:
        result = detect_app.detect_file(file_path, is_indoor, save_processed_images, False)
    except Exception as e:
        return ajax(0, f'Detect failed: {e}', None)

    coordinate_list = []

    for coordinate in result:
        _coordinate = [[int(x) for x in point] for point in coordinate]
        coordinate_list.append(_coordinate)

    print(coordinate_list)

    return ajax(1, 'ok', coordinate_list)