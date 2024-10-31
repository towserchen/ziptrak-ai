# Ziptrak AI openning detection

## Introduction

Use AI models to detect the positions of openings in indoor and outdoor scenes for the purpose of overlaying 3D models at those locations.

[Show Doc](docs/docs.md)

Author: Towser (towserchen#gmail.com)


## Usage

### Run Environment Requirement

Supports Windows, Linux and Mac, Python 3.10 or later must be installed. A server with a GPU/NPU is very helpful to promote the performance.

During installation, you may encounter various issues related to CUDA and torch. Please refer to the corresponding official documentation. Ideally, using a properly configured GPU that supports CUDA as the computational platform will yield the best performance.

### Install

1, Clone this repo, remember to run `git lfs pull` to pull all weight files. If github runs out of outbound traffic, download it manually from the following address:

`wget -O  "weights.zip" https://tmp-hd106.vx-cdn.com/file-6722577d87b1e-6722601f0d439/weights.zip`

2, Install all dependencies:

`pip install -r requriements.txt`

### Run

#### Command line

1, Adjust the app.py, change variable `file_path` to your image that want to be detected, and `is_window_detected`

2, `python app.py`

#### HTTP API

##### Launch API server

Modify params according to your requirements and run:

`uvicorn api:app --host 0.0.0.0 --port 80`

Further, you need to configure yourself reverse proxy and SSL certificates if the app runs in a production environment.

##### API list

Upload an image and detect it

POST `/detect`

**Params**

| Parameter              | Type   | Meaning                                      | Required | Default |
|------------------------|--------|----------------------------------------------|----------|---------|
| upload_file            | File   | The file to be uploaded, from HTML File | Yes      | None    |
| is_window_detected     | int    | Whether to detect windows, where 1 or 0 indicates True and False | No       | 1       |
| save_processed_images   | int    | Whether to save processed files, where 1 or 0 indicates True and False; if 1, it will save the object detection result images in the /results/ directory | No       | 0       |


## The basic principles of the app

This application combines YOLO-World and Efficient-SAM. It uses the zero-shot detection capability of YOLO-World to detect the contours of windows and pillars, and then utilizes ESAM for segmentation to obtain details. Finally, a post-processing function is applied to obtain the final result coordinates.


## Optimization and Improvement

## AI Model

YOLO-World is a workaround and not the most ideal solution. Training with YOLO on a well-annotated dataset will yield a more accurate model.

E-SAM shows good detection performance, and switching to SAM2 will not significantly improve (at least based on the current result set).

## API

The API code is provided only as an example. In a production environment, you will need to consider details such as authentication, rate limiting, and file upload management, and further customize it based on your specific scenario.

