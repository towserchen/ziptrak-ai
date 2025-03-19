import os
import time
import requests
from dotenv import load_dotenv
import boto3
from typing import Any

load_dotenv()

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000')
API_SECRET_KEY = os.environ.get('API_SECRET_KEY', 'ziptrak')

INSTANCE_ID = None


def get_instance_id():
    global INSTANCE_ID

    if INSTANCE_ID is None:
        metadata_url = 'http://169.254.169.254/latest/meta-data/instance-id'
        
        try:
            response = requests.get(metadata_url, timeout=1)
            if response.status_code == 200:
                INSTANCE_ID = response.text
                return response.text
            else:
                return None
        except requests.exceptions.RequestException as e:
            return None

    return INSTANCE_ID
    

def post(url: str, data: dict = {}):
    retryCount = 1
    retryMax = 5

    while retryCount < retryMax:
        response = requests.post(url, json=data)

        if response.status_code != 200:
            time.sleep(3)

            retryCount += 1
        else:
            break
    
    return response


# Report a detection result to gateway
def report(task_uuid: str, result: Any):
    data = {
        'instance_id': get_instance_id(),
        'task_uuid': task_uuid,
        'result': result
    }

    api_url = API_BASE_URL + '/task/report'
    post(api_url, data)



# Get a task from gateway
def get_task():
    api_url = API_BASE_URL + '/task/get'
    response = requests.get(api_url)

    if response.status_code == 200:
        if response.json()['status'] != 1:
            return None
        return response.json()['data']
    else:
        return None