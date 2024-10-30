# Ziptrak AI openning detection

## Introduction

Use AI models to detect the positions of openings in indoor and outdoor scenes for the purpose of overlaying 3D models at those locations.


## Usage

### Run Environment Requirement

Supports Windows, Linux and Mac, Python 3.10 or later must be installed. A server with a GPU/NPU is very helpful to promote the performance.

### Install

1, Clone this repo, remember to run `git lfs pull` to pull all weight files. If github runs out of outbound traffic, download it manually from the following address:
`wget -O  "weights.zip" https://tmp-hd106.vx-cdn.com/file-6722577d87b1e-6722601f0d439/weights.zip`

2, Install all dependencies:

`pip install -r requriements.txt`

3, Run the HTTP API server, and then you can configure your reverse proxy server:

`uvicorn app:app --host 0.0.0.0 --port 80`

### Run locally

### Run via API