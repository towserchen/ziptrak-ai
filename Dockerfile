FROM nvidia.cuda:12.5.1-cudnn-runtime-ubuntu22.04

ENV API_BASE_URL='http://127.0.0.1:8000'
ENV API_SECRET_KEY='ziptrak'

WORKDIR /app

COPY . /app

RUN apt update -y && apt-get install libgl1-mesa-glx libglib2.0-0 wget unzip python3 python3-pip -y && pip3 install -r requirements.txt

EXPOSE 10000

CMD ["python", "app.py"]
